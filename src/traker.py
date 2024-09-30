from abc import ABC, abstractmethod
from enum import Enum
import json
import math
import os
from typing import Iterable, Optional, Union
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import Tensor

import logging
import numpy as np
from numpy.lib.format import open_memmap
import torch
from torch import nn

def parameters_to_vector(parameters) -> Tensor:
    """
    Same as https://pytorch.org/docs/stable/generated/torch.nn.utils.parameters_to_vector.html
    but with :code:`reshape` instead of :code:`view` to avoid a pesky error.
    """
    vec = []
    for param in parameters:
        vec.append(param.reshape(-1))
    return torch.cat(vec)

def get_num_params(model: nn.Module) -> int:
    return parameters_to_vector(model.parameters()).numel()

def get_output_memory(features: Tensor, target_grads: Tensor, target_dtype: type):
    output_shape = features.size(0) * target_grads.size(0)
    output_dtype_size = torch.empty((1,), dtype=target_dtype).element_size()

    return output_shape * output_dtype_size


def get_free_memory(device):
    reserved = torch.cuda.memory_reserved(device=device)
    allocated = torch.cuda.memory_allocated(device=device)

    free = reserved - allocated
    return free


def get_matrix_mult_standard(
    features: Tensor, target_grads: Tensor, target_dtype: type
):
    output = features @ target_grads.t()
    return output.clone().to(target_dtype)


def get_matrix_mult_blockwise(
    features: Tensor, target_grads: Tensor, target_dtype: type, bs: int
):
    s_features = features.shape[0]
    s_target_grads = target_grads.shape[0]

    bs = min(s_features, s_target_grads, bs)

    # Copy the data in a pinned memory location to allow non-blocking
    # copies to the GPU
    features = features.pin_memory()
    target_grads = target_grads.pin_memory()

    # precompute all the blocks we will have to compute
    slices = []
    for i in range(int(np.ceil(s_features / bs))):
        for j in range(int(np.ceil(s_target_grads / bs))):
            slices.append((slice(i * bs, (i + 1) * bs), slice(j * bs, (j + 1) * bs)))

    # Allocate memory for the final output.
    final_output = torch.empty(
        (s_features, s_target_grads), dtype=target_dtype, device="cpu"
    )

    # Output buffers pinned on the CPU to be able to collect data from the
    # GPU asynchronously
    # For each of our (2) cuda streams we need two output buffer, one
    # is currently written on with the next batch of result and the
    # second one is already finished and getting copied on the final output

    # If the size is not a multiple of batch size we need extra buffers
    # with the proper shapes
    outputs = [
        torch.zeros((bs, bs), dtype=target_dtype, device=features.device).pin_memory()
        for x in range(4)
    ]
    left_bottom = s_features % bs
    options = [outputs]  # List of buffers we can potentially use
    if left_bottom:
        outputs_target_gradsottom = [
            torch.zeros(
                (left_bottom, bs), dtype=target_dtype, device=features.device
            ).pin_memory()
            for x in range(4)
        ]
        options.append(outputs_target_gradsottom)
    left_right = s_target_grads % bs
    if left_right:
        outputs_right = [
            torch.zeros(
                (bs, left_right), dtype=target_dtype, device=features.device
            ).pin_memory()
            for x in range(4)
        ]
        options.append(outputs_right)
    if left_right and left_bottom:
        outputs_corner = [
            torch.zeros(
                (left_bottom, left_right), dtype=target_dtype, device=features.device
            ).pin_memory()
            for x in range(4)
        ]
        options.append(outputs_corner)

    streams = [torch.cuda.Stream() for x in range(2)]

    # The slice that was computed last and need to now copied onto the
    # final output
    previous_slice = None

    def find_buffer_for_shape(shape):
        for buff in options:
            if buff[0].shape == shape:
                return buff
        return None

    for i, (slice_i, slice_j) in enumerate(slices):
        with torch.cuda.stream(streams[i % len(streams)]):
            # Copy the relevant blocks from CPU to the GPU asynchronously
            features_i = features[slice_i, :].cuda(non_blocking=True)
            target_grads_j = target_grads[slice_j, :].cuda(non_blocking=True)

            output_slice = features_i @ target_grads_j.t()

            find_buffer_for_shape(output_slice.shape)[i % 4].copy_(
                output_slice, non_blocking=False
            )

        # Write the previous batch of data from the temporary buffer
        # onto the final one (note that this was done by the other stream
        # so we swap back to the other one
        with torch.cuda.stream(streams[(i + 1) % len(streams)]):
            if previous_slice is not None:
                output_slice = final_output[previous_slice[0], previous_slice[1]]
                output_slice.copy_(
                    find_buffer_for_shape(output_slice.shape)[(i - 1) % 4],
                    non_blocking=True,
                )

        previous_slice = (slice_i, slice_j)

    # Wait for all the calculations/copies to be done
    torch.cuda.synchronize()

    # Copy the last chunk to the final result (from the appropriate buffer)
    output_slice = final_output[previous_slice[0], previous_slice[1]]
    output_slice.copy_(
        find_buffer_for_shape(output_slice.shape)[i % 4], non_blocking=True
    )

    return final_output


def get_matrix_mult(
    features: Tensor,
    target_grads: Tensor,
    target_dtype: torch.dtype = None,
    batch_size: int = 8096,
    use_blockwise: bool = False,
) -> Tensor:
    """

    Computes features @ target_grads.T. If the output matrix is too large to fit
    in memory, it will be computed in blocks.

    Args:
        features (Tensor):
            The first matrix to multiply.
        target_grads (Tensor):
            The second matrix to multiply.
        target_dtype (torch.dtype, optional):
            The dtype of the output matrix. If None, defaults to the dtype of
            features. Defaults to None.
        batch_size (int, optional):
            The batch size to use for blockwise matrix multiplication. Defaults
            to 8096.
        use_blockwise (bool, optional):
            Whether or not to use blockwise matrix multiplication. Defaults to
            False.

    """
    if target_dtype is None:
        target_dtype = features.dtype

    if use_blockwise:
        return get_matrix_mult_blockwise(
            features.cpu(), target_grads.cpu(), target_dtype, batch_size
        )
    elif features.device.type == "cpu":
        return get_matrix_mult_standard(features, target_grads, target_dtype)

    output_memory = get_output_memory(features, target_grads, target_dtype)
    free_memory = get_free_memory(features.device)

    if output_memory < free_memory:
        return get_matrix_mult_standard(features, target_grads, target_dtype)
    else:
        return get_matrix_mult_blockwise(
            features.cpu(), target_grads.cpu(), target_dtype, batch_size
        )

def vectorize(g, arr=None, device="cuda") -> Tensor:
    """
    records result into arr

    gradients are given as a dict :code:`(name_w0: grad_w0, ... name_wp:
    grad_wp)` where :code:`p` is the number of weight matrices. each
    :code:`grad_wi` has shape :code:`[batch_size, ...]` this function flattens
    :code:`g` to have shape :code:`[batch_size, num_params]`.
    """
    if arr is None:
        g_elt = g[list(g.keys())[0]]
        batch_size = g_elt.shape[0]
        num_params = 0
        for param in g.values():
            assert param.shape[0] == batch_size
            num_params += int(param.numel() / batch_size)
        arr = torch.empty(size=(batch_size, num_params), dtype=g_elt.dtype, device=device)

    pointer = 0
    for param in g.values():
        if len(param.shape) < 2:
            num_param = 1
            p = param.data.reshape(-1, 1)
        else:
            num_param = param[0].numel()
            p = param.flatten(start_dim=1).data

        arr[:, pointer : pointer + num_param] = p.to(device)
        pointer += num_param

    return arr


class ModelIDException(Exception):
    """A minimal custom exception for errors related to model IDs"""

    pass

class AbstractSaver(ABC):
    """
    Implementations of Saver class must implement getters and setters for TRAK
    features and scores, as well as intermediate values like gradients and
    "out-to-loss-gradient".

    The Saver class also handles the recording of metadata associated with each
    TRAK run. For example, hyperparameters like "JL dimension" -- the dimension
    used for the dimensionality reduction step of TRAK (Johnson-Lindenstrauss
    projection).
    """

    @abstractmethod
    def __init__(
        self,
        save_dir: Union[Path, str],
        metadata: Iterable,
        load_from_save_dir: bool,
        logging_level: int,
        use_half_precision: bool,
    ) -> None:
        """Creates the save directory if it doesn't already exist.
        If the save directory already exists, it validates that the current
        TRAKer class has the same hyperparameters (metadata) as the one
        specified in the save directory. Next, this method loads any existing
        computed results / intermediate values in the save directory. Last, it
        initalizes the self.current_store attributes which will be later
        populated with data for the "current" model ID of the TRAKer instance.

        Args:
            save_dir (Union[Path, str]): directory to save TRAK results,
                intermediate values, and metadata
            metadata (Iterable): a dictionary containing metadata related to the
                TRAKer class
            load_from_save_dir (bool): If True, the Saver instance will attempt
                to load existing metadata from save_dir. May lead to I/O issues
                if multiple Saver instances ran in parallel have this flag set
                to True. See the SLURM tutorial in our docs for more details.
            logging_level (int):
                logging level for the logger associated with this Saver instance
            use_half_precision (bool):
                If True, the Saver instance will save all results and intermediate
                values in half precision (float16).
        """
        self.metadata = metadata
        self.save_dir = Path(save_dir).resolve()
        self.load_from_save_dir = load_from_save_dir
        self.use_half_precision = use_half_precision

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir.joinpath("scores"), exist_ok=True)

        self.logger = logging.getLogger("STORE")
        self.logger.setLevel(logging_level)

        # init TRAKer metadata
        self.metadata_file = self.save_dir.joinpath("metadata.json")
        if os.path.exists(self.metadata_file) and self.load_from_save_dir:
            with open(self.metadata_file, "r") as f:
                existsing_metadata = json.load(f)
            existing_jl_dim = int(existsing_metadata["JL dimension"])
            assert (
                self.metadata["JL dimension"] == existing_jl_dim
            ), f"In {self.save_dir} there are models using JL dimension {existing_jl_dim},\n\
                   and this TRAKer instance uses JL dimension {self.metadata['JL dimension']}."

            existing_matrix_type = existsing_metadata["JL matrix type"]
            assert (
                self.metadata["JL matrix type"] == existing_matrix_type
            ), f"In {self.save_dir} there are models using a {existing_matrix_type} JL matrix,\n\
                   and this TRAKer instance uses a {self.metadata['JL matrix type']} JL matrix."

            assert (
                self.metadata["train set size"] == existsing_metadata["train set size"]
            ), f"In {self.save_dir} there are models TRAKing\n\
                   {existsing_metadata['train set size']} examples, and in this TRAKer instance\n\
                   there are {self.metadata['train set size']} examples."

        elif self.load_from_save_dir:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f)

        self.model_ids = {}
        self.experiments = {}
        self.experiments_file = self.save_dir.joinpath("experiments.json")
        if self.load_from_save_dir:
            # check if there are existing model ids in the save_dir
            self.model_ids_files = self.save_dir.rglob("id_*.json")

            for existing_model_id_file in self.model_ids_files:
                with open(existing_model_id_file, "r") as f:
                    existing_id = json.load(f)
                    existing_id = {
                        int(model_id): metadata
                        for model_id, metadata in existing_id.items()
                    }
                self.model_ids.update(existing_id)

            if os.path.isfile(self.experiments_file):
                with open(self.experiments_file, "r") as f:
                    self.experiments.update(json.load(f))
            else:
                with open(self.experiments_file, "w") as f:
                    json.dump({}, f)

        existing_ids = list(self.model_ids.keys())
        if len(existing_ids) > 0:
            self.logger.info(
                f"Existing model IDs in {self.save_dir}: {sorted(existing_ids)}"
            )
            ids_finalized = sorted(
                list([id for id, v in self.model_ids.items() if v["is_finalized"] == 1])
            )
            if len(ids_finalized) > 0:
                self.logger.info(f"Model IDs that have been finalized: {ids_finalized}")
            else:
                self.logger.info(
                    f"No model IDs in {self.save_dir} have been finalized."
                )
        else:
            self.logger.info(f"No existing model IDs in {self.save_dir}.")

        if len(list(self.experiments.keys())) > 0:
            self.logger.info("Existing TRAK scores:")
            for exp_name, values in self.experiments.items():
                self.logger.info(f"{exp_name}: {values['scores_path']}")
        else:
            self.logger.info(f"No existing TRAK scores in {self.save_dir}.")

        self.current_model_id = None
        self.current_store = {
            "grads": None,
            "out_to_loss": None,
            "features": None,
        }

    @abstractmethod
    def register_model_id(self, model_id: int) -> None:
        """Create metadata for a new model ID (checkpoint).

        Args:
            model_id (int):
                a unique ID for a checkpoint

        """
        ...

    @abstractmethod
    def serialize_current_model_id_metadata(self) -> None:
        """Write to disk / commit any updates to the metadata associated
        to the current model ID

        """
        ...

    @abstractmethod
    def init_store(self, model_id: int) -> None:
        """Initializes store for a given model ID (checkpoint).

        Args:
            model_id (int):
                a unique ID for a checkpoint
        """
        ...

    @abstractmethod
    def init_experiment(self, model_id: int) -> None:
        """Initializes store for a given experiment & model ID (checkpoint).

        Args:
            model_id (int):
                a unique ID for a checkpoint
        """
        ...

    @abstractmethod
    def load_current_store(self, model_id: int) -> None:
        """Populates the self.current_store attributes with data for the
        given model ID (checkpoint).

        Args:
            model_id (int):
                a unique ID for a checkpoint

        """
        ...

    @abstractmethod
    def save_scores(self, exp_name: str) -> None:
        """Saves scores for a given experiment name

        Args:
            exp_name (str):
                experiment name

        """
        ...

    @abstractmethod
    def del_grads(self, model_id: int, target: bool) -> None:
        """Delete the intermediate values (gradients) for a given model id

        Args:
            model_id (int):
                a unique ID for a checkpoint
            target (bool):
                if True, delete the gradients of the target samples, otherwise
                delete the train set gradients.

        """
        ...

class MmapSaver(AbstractSaver):
    """A saver that uses memory-mapped numpy arrays. This makes small reads and
    writes (e.g.) during featurizing feasible without loading the entire file
    into memory.

    """

    def __init__(
        self,
        save_dir,
        metadata,
        train_set_size,
        proj_dim,
        load_from_save_dir,
        logging_level,
        use_half_precision,
    ) -> None:
        super().__init__(
            save_dir=save_dir,
            metadata=metadata,
            load_from_save_dir=load_from_save_dir,
            logging_level=logging_level,
            use_half_precision=use_half_precision,
        )
        self.train_set_size = train_set_size
        self.proj_dim = proj_dim

    def register_model_id(
        self, model_id: int, _allow_featurizing_already_registered: bool
    ) -> None:
        """This method
        1) checks if the model ID already exists in the save dir
        2) if yes, it raises an error since model IDs must be unique
        3) if not, it creates a metadata file for it and initalizes store mmaps

        Args:
            model_id (int):
                a unique ID for a checkpoint

        Raises:
            ModelIDException:
                raised if the model ID to be registered already exists

        """
        self.current_model_id = model_id

        if self.current_model_id in self.model_ids.keys() and (
            not _allow_featurizing_already_registered
        ):
            err_msg = f"model id {self.current_model_id} is already registered. Check {self.save_dir}"
            raise ModelIDException(err_msg)
        self.model_ids[self.current_model_id] = {"is_featurized": 0, "is_finalized": 0}

        self.init_store(self.current_model_id)
        self.serialize_current_model_id_metadata(already_exists=False)

    def serialize_current_model_id_metadata(self, already_exists=True) -> None:
        is_featurized = int(
            self.current_store["is_featurized"].sum() == self.train_set_size
        )

        # update the metadata JSON file
        content = {
            self.current_model_id: {
                "is_featurized": is_featurized,
                "is_finalized": self.model_ids[self.current_model_id]["is_finalized"],
            }
        }
        # update the metadata dict within the class instance
        self.model_ids[self.current_model_id]["is_featurized"] = is_featurized
        if (is_featurized == 1) or not already_exists:
            with open(
                self.save_dir.joinpath(f"id_{self.current_model_id}.json"), "w"
            ) as f:
                json.dump(content, f)

    def init_store(self, model_id) -> None:
        prefix = self.save_dir.joinpath(str(model_id))
        if os.path.exists(prefix):
            self.logger.info(f"Model ID folder {prefix} already exists")
        os.makedirs(prefix, exist_ok=True)
        featurized_so_far = np.zeros(shape=(self.train_set_size,), dtype=np.int32)
        ft = self._load(
            prefix.joinpath("_is_featurized.mmap"),
            shape=(self.train_set_size,),
            mode="w+",
            dtype=np.int32,
        )
        ft[:] = featurized_so_far[:]
        ft.flush()

        self.load_current_store(model_id, mode="w+")

    def init_experiment(self, exp_name, num_targets, model_id) -> None:
        prefix = self.save_dir.joinpath(str(model_id))
        if not os.path.exists(prefix):
            raise ModelIDException(
                f"model ID folder {prefix} does not exist,\n\
            cannot start scoring"
            )
        self.experiments[exp_name] = {
            "num_targets": num_targets,
            "scores_path": str(self.save_dir.joinpath(f"scores/{exp_name}.mmap")),
            "scores_finalized": 0,
        }

        # update experiments.json
        with open(self.experiments_file, "r") as fp:
            exp_f = json.load(fp)

        exp_f[exp_name] = self.experiments[exp_name]
        with open(self.experiments_file, "w") as fp:
            json.dump(exp_f, fp)

        if os.path.exists(prefix.joinpath(f"{exp_name}_grads.mmap")):
            mode = "r+"
        else:
            mode = "w+"
        self.load_current_store(
            model_id=model_id, exp_name=exp_name, exp_num_targets=num_targets, mode=mode
        )

    def _load(self, fname, shape, mode, dtype=None):
        if mode == "w+":
            self.logger.debug(f"Creating {fname}.")
        else:
            self.logger.debug(f"Loading {fname}.")
        if dtype is None:
            dtype = np.float16 if self.use_half_precision else np.float32
        try:
            return open_memmap(filename=fname, mode=mode, shape=shape, dtype=dtype)
        except OSError:
            self.logger.info(f"{fname} does not exist, skipping.")
            return None

    def load_current_store(
        self,
        model_id: int,
        exp_name: Optional[str] = None,
        exp_num_targets: Optional[int] = -1,
        mode: Optional[str] = "r+",
    ) -> None:
        """This method uses numpy memmaps for serializing the TRAK results and
        intermediate values.

        Args:
            model_id (int):
                a unique ID for a checkpoint
            exp_name (str, optional):
                Experiment name for which to load the features. If None, loads
                the train (source) features for a model ID. Defaults to None.
            exp_num_targets (int, optional):
                Number of targets for the experiment. Specify only when exp_name
                is not None. Defaults to -1.
            mode (str, optional):
                Defaults to 'r+'.

        """
        self.current_model_id = model_id
        if exp_name is not None:
            self.current_experiment_name = exp_name
        prefix = self.save_dir.joinpath(str(self.current_model_id))

        if exp_name is None:
            to_load = {
                "grads": (
                    prefix.joinpath("grads.mmap"),
                    (self.train_set_size, self.proj_dim),
                    None,
                ),
                "out_to_loss": (
                    prefix.joinpath("out_to_loss.mmap"),
                    (self.train_set_size, 1),
                    None,
                ),
                "features": (
                    prefix.joinpath("features.mmap"),
                    (self.train_set_size, self.proj_dim),
                    None,
                ),
                "is_featurized": (
                    prefix.joinpath("_is_featurized.mmap"),
                    (self.train_set_size, 1),
                    np.int32,
                ),
            }
        else:
            to_load = {
                f"{exp_name}_grads": (
                    prefix.joinpath(f"{exp_name}_grads.mmap"),
                    (exp_num_targets, self.proj_dim),
                    None,
                ),
                f"{exp_name}_scores": (
                    self.save_dir.joinpath(f"scores/{exp_name}.mmap"),
                    (self.train_set_size, exp_num_targets),
                    None,
                ),
            }

        for name, (path, shape, dtype) in to_load.items():
            self.current_store[name] = self._load(path, shape, mode, dtype)

    def save_scores(self, exp_name):
        assert self.current_experiment_name == exp_name
        prefix = self.save_dir.joinpath("scores")
        self.logger.info(f"Saving scores in {prefix}/{exp_name}.mmap")
        self.current_store[f"{exp_name}_scores"].flush()
        self.experiments[exp_name]["scores_finalized"] = 1
        with open(self.experiments_file, "w") as fp:
            json.dump(self.experiments, fp)

    def del_grads(self, model_id):
        grads_file = self.save_dir.joinpath(str(model_id)).joinpath("grads.mmap")

        # delete grads memmap
        grads_file.unlink()


class ModelIDException(Exception):
    """A minimal custom exception for errors related to model IDs"""

    pass

class AbstractModelOutput(ABC):
    """See, e.g. `this tutorial <https://trak.readthedocs.io/en/latest/clip.html>`_
    for an example on how to subclass :code:`AbstractModelOutput` for a task of
    your choice.

    Subclasses must implement:

    - a :code:`get_output` method that takes in a batch of inputs and model
      weights to produce outputs that TRAK will be trained to predict. In the
      notation of the paper, :code:`get_output` should return :math:`f(z,\\theta)`

    - a :code:`get_out_to_loss_grad` method that takes in a batch of inputs and
      model weights to produce the gradient of the function that transforms the
      model outputs above into the loss with respect to the batch. In the
      notation of the paper, :code:`get_out_to_loss_grad` returns (entries along
      the diagonal of) :math:`Q`.

    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_output(self, model, batch: Iterable[Tensor]) -> Tensor:
        """See Sections 2 & 3 of `our paper
        <https://arxiv.org/abs/2303.14186>`_ for more details on what model
        output functions are in the context of TRAK and how to use & design
        them.

        Args:
            model (torch.nn.Module):
                model
            batch (Iterable[Tensor]):
                input batch

        Returns:
            Tensor:
                model output function
        """
        ...

    @abstractmethod
    def get_out_to_loss_grad(self, model, batch: Iterable[Tensor]) -> Tensor:
        """See Sections 2 & 3 of `our paper
        <https://arxiv.org/abs/2303.14186>`_ for more details on what the
        out-to-loss functions (in the notation of the paper, :math:`Q`) are in
        the context of TRAK and how to use & design them.

        Args:
            model (torch.nn.Module): model
            batch (Iterable[Tensor]): input batch

        Returns:
            Tensor: gradient of the out-to-loss function
        """
        ...


class ImageClassificationModelOutput(AbstractModelOutput):
    """Margin for (multiclass) image classification. See Section 3.3 of `our
    paper <https://arxiv.org/abs/2303.14186>`_ for more details.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """
        Args:
            temperature (float, optional): Temperature to use inside the
            softmax for the out-to-loss function. Defaults to 1.
        """
        super().__init__()
        self.softmax = nn.Softmax(-1)
        self.loss_temperature = temperature

    @staticmethod
    def get_output(
        model: nn.Module,
        weights: Iterable[Tensor],
        buffers: Iterable[Tensor],
        image: Tensor,
        label: Tensor,
    ) -> Tensor:
        """For a given input :math:`z=(x, y)` and model parameters :math:`\\theta`,
        let :math:`p(z, \\theta)` be the softmax probability of the correct class.
        This method implements the model output function

        .. math::

            \\log(\\frac{p(z, \\theta)}{1 - p(z, \\theta)}).

        It uses functional models from torch.func (previously functorch) to make
        the per-sample gradient computations (much) faster. For more details on
        what functional models are, and how to use them, please refer to
        https://pytorch.org/docs/stable/func.html and
        https://pytorch.org/functorch/stable/notebooks/per_sample_grads.html.

        Args:
            model (torch.nn.Module):
                torch model
            weights (Iterable[Tensor]):
                functorch model weights
            buffers (Iterable[Tensor]):
                functorch model buffers
            image (Tensor):
                input image, should not have batch dimension
            label (Tensor):
                input label, should not have batch dimension

        Returns:
            Tensor:
                model output for the given image-label pair :math:`z` and
                weights & buffers :math:`\\theta`.
        """
        logits = torch.func.functional_call(model, (weights, buffers), image.unsqueeze(0))
        bindex = torch.arange(logits.shape[0]).to(logits.device, non_blocking=False)
        logits_correct = logits[bindex, label.unsqueeze(0)]

        cloned_logits = logits.clone()
        # remove the logits of the correct labels from the sum
        # in logsumexp by setting to -torch.inf
        cloned_logits[bindex, label.unsqueeze(0)] = torch.tensor(
            -torch.inf, device=logits.device, dtype=logits.dtype
        )

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins.sum()

    def get_out_to_loss_grad(
        self, model, weights, buffers, batch: Iterable[Tensor]
    ) -> Tensor:
        """Computes the (reweighting term Q in the paper)

        Args:
            model (torch.nn.Module):
                torch model
            weights (Iterable[Tensor]):
                functorch model weights
            buffers (Iterable[Tensor]):
                functorch model buffers
            batch (Iterable[Tensor]):
                input batch

        Returns:
            Tensor:
                out-to-loss (reweighting term) for the input batch
        """
        images, labels = batch
        logits = torch.func.functional_call(model, (weights, buffers), images)
        # here we are directly implementing the gradient instead of relying on autodiff to do
        # that for us
        ps = self.softmax(logits / self.loss_temperature)[
            torch.arange(logits.size(0)), labels
        ]
        return (1 - ps).clone().detach().unsqueeze(-1)


class ProjectionType(str, Enum):
    normal: str = "normal"
    rademacher: str = "rademacher"


class AbstractProjector(ABC):
    """Implementations of the Projector class must implement the
    :meth:`AbstractProjector.project` method, which takes in model gradients and
    returns
    """

    @abstractmethod
    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: Union[str, ProjectionType],
        device: Union[str, torch.device],
    ) -> None:
        """Initializes hyperparameters for the projection.

        Args:
            grad_dim (int):
                number of parameters in the model (dimension of the gradient
                vectors)
            proj_dim (int):
                dimension after the projection
            seed (int):
                random seed for the generation of the sketching (projection)
                matrix
            proj_type (Union[str, ProjectionType]):
                the random projection (JL transform) guearantees that distances
                will be approximately preserved for a variety of choices of the
                random matrix (see e.g. https://arxiv.org/abs/1411.2404). Here,
                we provide an implementation for matrices with iid Gaussian
                entries and iid Rademacher entries.
            device (Union[str, torch.device]):
                CUDA device to use

        """
        self.grad_dim = grad_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.proj_type = proj_type
        self.device = device

    @abstractmethod
    def project(self, grads: Tensor, model_id: int) -> Tensor:
        """Performs the random projection. Model ID is included
        so that we generate different projection matrices for every
        model ID.

        Args:
            grads (Tensor): a batch of gradients to be projected
            model_id (int): a unique ID for a checkpoint

        Returns:
            Tensor: the projected gradients
        """

    def free_memory(self):
        """Frees up memory used by the projector."""

class NoOpProjector(AbstractProjector):
    """
    A projector that returns the gradients as they are, i.e., implements
    :code:`projector.project(grad) = grad`.
    """

    def __init__(
        self,
        grad_dim: int = 0,
        proj_dim: int = 0,
        seed: int = 0,
        proj_type: Union[str, ProjectionType] = "na",
        device: Union[str, torch.device] = "cuda",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

    def project(self, grads: Tensor, model_id: int) -> Tensor:
        """A no-op method.

        Args:
            grads (Tensor): a batch of gradients to be projected
            model_id (int): a unique ID for a checkpoint

        Returns:
            Tensor: the (non-)projected gradients
        """
        if isinstance(grads, dict):
            grads = vectorize(grads, device=self.device)
        return grads

    def free_memory(self):
        """A no-op method."""
        pass

class BasicProjector(AbstractProjector):
    """
    A simple block-wise implementation of the projection. The projection matrix
    is generated on-device in blocks. The accumulated result across blocks is
    returned.

    Note: This class will be significantly slower and have a larger memory
    footprint than the CudaProjector. It is recommended that you use this method
    only if the CudaProjector is not available to you -- e.g. if you don't have
    a CUDA-enabled device with compute capability >=7.0 (see
    https://developer.nvidia.com/cuda-gpus).
    """

    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: ProjectionType,
        device: torch.device,
        block_size: int = 100,
        dtype: torch.dtype = torch.float32,
        model_id=0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

        self.block_size = min(self.proj_dim, block_size)
        self.num_blocks = math.ceil(self.proj_dim / self.block_size)
        self.dtype = dtype
        self.proj_type = proj_type
        self.model_id = model_id

        self.proj_matrix = torch.empty(
            self.grad_dim, self.block_size, dtype=self.dtype, device=self.device
        )

        self.proj_matrix_available = True

        self.generator = torch.Generator(device=self.device)

        self.get_generator_states()
        self.generate_sketch_matrix(self.generator_states[0])

    def free_memory(self):
        try:
            del self.proj_matrix
        except AttributeError:
            pass
        self.proj_matrix_available = False

    def get_generator_states(self):
        self.generator_states = []
        self.seeds = []
        self.jl_size = self.grad_dim * self.block_size

        for i in range(self.num_blocks):
            s = self.seed + int(1e3) * i + int(1e5) * self.model_id
            self.seeds.append(s)
            self.generator = self.generator.manual_seed(s)
            self.generator_states.append(self.generator.get_state())

    def generate_sketch_matrix(self, generator_state):
        if not self.proj_matrix_available:
            self.proj_matrix = torch.empty(
                self.grad_dim, self.block_size, dtype=self.dtype, device=self.device
            )
            self.proj_matrix_available = True

        self.generator.set_state(generator_state)
        if self.proj_type == ProjectionType.normal or self.proj_type == "normal":
            self.proj_matrix.normal_(generator=self.generator)
        elif (
            self.proj_type == ProjectionType.rademacher
            or self.proj_type == "rademacher"
        ):
            self.proj_matrix.bernoulli_(p=0.5, generator=self.generator)
            self.proj_matrix *= 2.0
            self.proj_matrix -= 1.0
        else:
            raise KeyError(f"Projection type {self.proj_type} not recognized.")

    def project(self, grads: Tensor, model_id: int) -> Tensor:
        if isinstance(grads, dict):
            grads = vectorize(grads, device=self.device)
        grads = grads.to(dtype=self.dtype)
        sketch = torch.zeros(
            size=(grads.size(0), self.proj_dim), dtype=self.dtype, device=self.device
        )

        if model_id != self.model_id:
            self.model_id = model_id
            self.get_generator_states()  # regenerate random seeds for new model_id
            if self.num_blocks == 1:
                self.generate_sketch_matrix(self.generator_states[0])

        if self.num_blocks == 1:
            torch.matmul(grads.data, self.proj_matrix, out=sketch)
        else:
            for ind in range(self.num_blocks):
                self.generate_sketch_matrix(self.generator_states[ind])

                st = ind * self.block_size
                ed = min((ind + 1) * self.block_size, self.proj_dim)
                sketch[:, st:ed] = (
                    grads.type(self.dtype) @ self.proj_matrix[:, : (ed - st)]
                )
        return sketch.type(grads.dtype)

class CudaProjector(AbstractProjector):
    """
    A performant implementation of the projection for CUDA with compute
    capability >= 7.0.
    """

    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: ProjectionType,
        device,
        max_batch_size: int,
        *args,
        **kwargs,
    ) -> None:
        """

        Args:
            grad_dim (int):
                Number of parameters
            proj_dim (int):
                Dimension we project *to* during the projection step
            seed (int):
                Random seed
            proj_type (ProjectionType):
                Type of randomness to use for projection matrix (rademacher or normal)
            device:
                CUDA device
            max_batch_size (int):
                Explicitly constraints the batch size the CudaProjector is going
                to use for projection. Set this if you get a 'The batch size of
                the CudaProjector is too large for your GPU' error. Must be
                either 8, 16, or 32.

        Raises:
            ValueError:
                When attempting to use this on a non-CUDA device
            ModuleNotFoundError:
                When fast_jl is not installed

        """
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)
        self.max_batch_size = max_batch_size

        if isinstance(device, str):
            device = torch.device(device)

        if device.type != "cuda":
            err = "CudaProjector only works on a CUDA device; Either switch to a CUDA device, or use the BasicProjector"
            raise ValueError(err)

        self.num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count

        try:
            import fast_jl

            # test run to catch at init time if projection goes through
            fast_jl.project_rademacher_8(
                torch.zeros(8, 1_000, device="cuda"), 512, 0, self.num_sms
            )
        except ImportError:
            err = "You should make sure to install the CUDA projector for traker (called fast_jl).\
                  See the installation FAQs for more details."
            raise ModuleNotFoundError(err)

    def project(
        self,
        grads: Union[dict, Tensor],
        model_id: int,
    ) -> Tensor:
        if isinstance(grads, dict):
            grads = vectorize(grads, device=self.device)

        batch_size = grads.shape[0]

        effective_batch_size = 32
        if batch_size <= 8:
            effective_batch_size = 8
        elif batch_size <= 16:
            effective_batch_size = 16

        effective_batch_size = min(self.max_batch_size, effective_batch_size)

        function_name = f"project_{self.proj_type.value}_{effective_batch_size}"
        import fast_jl

        fn = getattr(fast_jl, function_name)

        try:
            result = fn(
                grads, self.proj_dim, self.seed + int(1e4) * model_id, self.num_sms
            )
        except RuntimeError as e:
            if "CUDA error: too many resources requested for launch" in str(e):
                # provide a more helpful error message
                raise RuntimeError(
                    (
                        "The batch size of the CudaProjector is too large for your GPU. "
                        "Reduce it by using the proj_max_batch_size argument of the TRAKer.\nOriginal error:"
                    )
                )
            else:
                raise e

        return result

    def free_memory(self):
        """A no-op method."""
        pass

def get_parameter_chunk_sizes(
    model: nn.Module,
    batch_size: int,
):
    """The :class:`CudaProjector` supports projecting when the product of the
    number of parameters and the batch size is less than the the max value of
    int32. This function computes the number of parameters that can be projected
    at once for a given model and batch size.

    The method returns a tuple containing the maximum number of parameters that
    can be projected at once and a list of the actual number of parameters in
    each chunk (a sequence of paramter groups).  Used in
    :class:`ChunkedCudaProjector`.
    """
    param_shapes = []
    for p in model.parameters():
        param_shapes.append(p.numel())

    param_shapes = np.array(param_shapes)

    chunk_sum = 0
    max_chunk_size = np.iinfo(np.uint32).max // batch_size
    params_per_chunk = []

    for ps in param_shapes:
        if chunk_sum + ps >= max_chunk_size:
            params_per_chunk.append(chunk_sum)
            chunk_sum = 0

        chunk_sum += ps

    if param_shapes.sum() - np.sum(params_per_chunk) > 0:
        params_per_chunk.append(param_shapes.sum() - np.sum(params_per_chunk))

    return max_chunk_size, params_per_chunk

class ChunkedCudaProjector:
    def __init__(
        self,
        projector_per_chunk: list,
        max_chunk_size: int,
        params_per_chunk: list,
        feat_bs: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.projector_per_chunk = projector_per_chunk
        self.proj_dim = self.projector_per_chunk[0].proj_dim
        self.proj_type = self.projector_per_chunk[0].proj_type
        self.params_per_chunk = params_per_chunk

        self.max_chunk_size = max_chunk_size
        self.feat_bs = feat_bs
        self.device = device
        self.dtype = dtype
        self.input_allocated = False

    def allocate_input(self):
        if self.input_allocated:
            return

        self.ch_input = torch.zeros(
            size=(self.feat_bs, self.max_chunk_size),
            device=self.device,
            dtype=self.dtype,
        )

        self.input_allocated = True

    def free_memory(self):
        if not self.input_allocated:
            return

        del self.ch_input
        self.input_allocated = False

    def project(self, grads, model_id):
        self.allocate_input()
        ch_output = torch.zeros(
            size=(self.feat_bs, self.proj_dim), device=self.device, dtype=self.dtype
        )
        pointer = 0
        # iterate over params, keep a counter of params so far, and when prev
        # chunk reaches max_chunk_size, project and accumulate
        projector_index = 0
        for i, p in enumerate(grads.values()):
            if len(p.shape) < 2:
                p_flat = p.data.unsqueeze(-1)
            else:
                p_flat = p.data.flatten(start_dim=1)

            param_size = p_flat.size(1)
            if pointer + param_size > self.max_chunk_size:
                # fill remaining entries with 0
                assert pointer == self.params_per_chunk[projector_index]
                # project and accumulate
                ch_output.add_(
                    self.projector_per_chunk[projector_index].project(
                        self.ch_input[:, :pointer].contiguous(),
                        model_id=model_id,
                    )
                )
                # reset counter
                pointer = 0
                projector_index += 1

            # continue accumulation
            actual_bs = min(self.ch_input.size(0), p_flat.size(0))
            self.ch_input[:actual_bs, pointer : pointer + param_size].copy_(p_flat)
            pointer += param_size

        # at the end, we need to project remaining items
        # fill remaining entries with 0
        assert pointer == self.params_per_chunk[projector_index]
        # project and accumulate
        ch_output[:actual_bs].add_(
            self.projector_per_chunk[projector_index].project(
                self.ch_input[:actual_bs, :pointer].contiguous(),
                model_id=model_id,
            )
        )

        return ch_output[:actual_bs]

class AbstractScoreComputer(ABC):
    """
    The :code:`ScoreComputer` class
    Implementations of the ScoreComputer class must implement three methods:
    - :code:`get_xtx`
    - :code:`get_x_xtx_inv`
    - :code:`get_scores`
    """

    @abstractmethod
    def __init__(self, dtype, device) -> None:
        self.dtype = dtype
        self.device = device

    @abstractmethod
    def get_xtx(self, grads: Tensor) -> Tensor:
        """Computes :math:`X^\top X`, where :math:`X` is the matrix of projected
        gradients. Here, the shape of :math:`X` is :code:`(n, p)`, where
        :math:`n` is the number of training examples and :math:`p` is the
        dimension of the projection.


        Args:
            grads (Tensor): projected gradients of shape :code:`(n, p)`.

        Returns:
            Tensor: :math:`X^\top X` of shape :code:`(p, p)`.
        """

    @abstractmethod
    def get_x_xtx_inv(self, grads: Tensor, xtx: Tensor) -> Tensor:
        """Computes :math:`X(X^\top X)^{-1}`, where :math:`X` is the matrix of
        projected gradients. Here, the shape of :math:`X` is :code:`(n, p)`,
        where :math:`n` is the number of training examples and :math:`p` is the
        dimension of the projection. This function takes as input the
        pre-computed :math:`X^\top X` matrix, which is computed by the
        :code:`get_xtx` method.

        Args:
            grads (Tensor): projected gradients :math:`X` of shape :code:`(n, p)`.
            xtx (Tensor): :math:`X^\top X` of shape :code:`(p, p)`.

        Returns:
            Tensor: :math:`X(X^\top X)^{-1}` of shape :code:`(n, p)`.
        """

    @abstractmethod
    def get_scores(
        self, features: Tensor, target_grads: Tensor, accumulator: Tensor
    ) -> None:
        """Computes the scores for a given set of features and target gradients.
        In particular, this function takes in a matrix of features
        :math:`\Phi=X(X^\top X)^{-1}`, computed by the :code:`get_x_xtx_inv`
        method, and a matrix of target (projected) gradients :math:`X_{target}`.
        Then, it computes the scores as :math:`\Phi X_{target}^\top`.  The
        resulting matrix has shape :code:`(n, m)`, where :math:`n` is the number
        of training examples and :math:`m` is the number of target examples.

        The :code:`accumulator` argument is used to store the result of the
        computation. This is useful when computing scores for multiple model
        checkpoints, as it allows us to re-use the same memory for the score
        matrix.

        Args:
            features (Tensor): features :math:`\Phi` of shape :code:`(n, p)`.
            target_grads (Tensor):
                target projected gradients :math:`X_{target}` of shape
                :code:`(m, p)`.
            accumulator (Tensor): accumulator of shape :code:`(n, m)`.
        """


class BasicSingleBlockScoreComputer(AbstractScoreComputer):
    """A bare-bones implementation of :code:`ScoreComputer` that will likely
    OOM for almost all applications. Here for testing purposes only. Unless you
    have a good reason not to, you should use :func:`BasicScoreComputer`
    instead.
    """

    def get_xtx(self, grads: Tensor) -> Tensor:
        return grads.T @ grads

    def get_x_xtx_inv(self, grads: Tensor, xtx: Tensor) -> Tensor:
        # torch.linalg.inv does not support float16
        return grads @ torch.linalg.inv(xtx.float()).to(self.dtype)

    def get_scores(
        self, features: Tensor, target_grads: Tensor, accumulator: Tensor
    ) -> None:
        accumulator += (features @ target_grads.T).detach().cpu()


class BasicScoreComputer(AbstractScoreComputer):
    """An implementation of :code:`ScoreComputer` that computes matmuls in a
    block-wise manner.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        CUDA_MAX_DIM_SIZE: int = 20_000,
        logging_level=logging.INFO,
        lambda_reg: float = 0.0,
    ) -> None:
        """
        Args:
            dtype (torch.dtype):
            device (Union[str, torch.device]):
                torch device to do matmuls on
            CUDA_MAX_DIM_SIZE (int, optional):
                Size of block for block-wise matmuls. Defaults to 100_000.
            logging_level (logging level, optional):
                Logging level for the logger. Defaults to logging.info.
            lambda_reg (int):
                regularization term for l2 reg on xtx
        """
        super().__init__(dtype, device)
        self.CUDA_MAX_DIM_SIZE = CUDA_MAX_DIM_SIZE
        self.logger = logging.getLogger("ScoreComputer")
        self.logger.setLevel(logging_level)
        self.lambda_reg = lambda_reg

    def get_xtx(self, grads: Tensor) -> Tensor:
        self.proj_dim = grads.shape[1]
        result = torch.zeros(
            self.proj_dim, self.proj_dim, dtype=self.dtype, device=self.device
        )
        blocks = torch.split(grads, split_size_or_sections=self.CUDA_MAX_DIM_SIZE, dim=0)

        for block in blocks:
            result += block.T.to(self.device) @ block.to(self.device)

        return result

    def get_x_xtx_inv(self, grads: Tensor, xtx: Tensor) -> Tensor:
        blocks = torch.split(grads, split_size_or_sections=self.CUDA_MAX_DIM_SIZE, dim=0)

        xtx_reg = xtx + self.lambda_reg * torch.eye(
            xtx.size(dim=0), device=xtx.device, dtype=xtx.dtype
        )
        xtx_inv = torch.linalg.inv(xtx_reg.to(torch.float32))

        # center X^TX inverse a bit to avoid numerical issues when going to float16
        xtx_inv /= xtx_inv.abs().mean()

        xtx_inv = xtx_inv.to(self.dtype)

        result = torch.empty(
            grads.shape[0], xtx_inv.shape[1], dtype=self.dtype, device=self.device
        )
        for i, block in enumerate(blocks):
            start = i * self.CUDA_MAX_DIM_SIZE
            end = min(grads.shape[0], (i + 1) * self.CUDA_MAX_DIM_SIZE)
            result[start:end] = block.to(self.device) @ xtx_inv
        return result

    def get_scores(
        self, features: Tensor, target_grads: Tensor, accumulator: Tensor
    ) -> Tensor:
        train_dim = features.shape[0]
        target_dim = target_grads.shape[0]

        self.logger.debug(f"{train_dim=}, {target_dim=}")

        accumulator += (
            get_matrix_mult(features=features, target_grads=target_grads).detach().cpu()
        )

class AbstractGradientComputer(ABC):
    """Implementations of the GradientComputer class should allow for
    per-sample gradients.  This is behavior is enabled with three methods:

    - the :meth:`.load_model_params` method, well, loads model parameters. It can
      be as simple as a :code:`self.model.load_state_dict(..)`

    - the :meth:`.compute_per_sample_grad` method computes per-sample gradients
      of the chosen model output function with respect to the model's parameters.

    - the :meth:`.compute_loss_grad` method computes the gradients of the loss
      function with respect to the model output (which should be a scalar) for
      every sample.

    """

    @abstractmethod
    def __init__(
        self,
        model: nn.Module,
        task: AbstractModelOutput,
        grad_dim: Optional[int] = None,
        dtype: Optional[torch.dtype] = torch.float16,
        device: Optional[torch.device] = "cuda",
    ) -> None:
        """Initializes attributes, nothing too interesting happening.

        Args:
            model (torch.nn.Module):
                model
            task (AbstractModelOutput):
                task (model output function)
            grad_dim (int, optional):
                Size of the gradients (number of model parameters). Defaults to
                None.
            dtype (torch.dtype, optional):
                Torch dtype of the gradients. Defaults to torch.float16.
            device (torch.device, optional):
                Torch device where gradients will be stored. Defaults to 'cuda'.

        """
        self.model = model
        self.modelout_fn = task
        self.grad_dim = grad_dim
        self.dtype = dtype
        self.device = device

    @abstractmethod
    def load_model_params(self, model) -> None:
        ...

    @abstractmethod
    def compute_per_sample_grad(self, batch: Iterable[Tensor]) -> Tensor:
        ...

    @abstractmethod
    def compute_loss_grad(self, batch: Iterable[Tensor], batch_size: int) -> Tensor:
        ...


class FunctionalGradientComputer(AbstractGradientComputer):
    def __init__(
        self,
        model: nn.Module,
        task: AbstractModelOutput,
        grad_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        grad_wrt: Optional[Iterable[str]] = None,
    ) -> None:
        """Initializes attributes, and loads model parameters.

        Args:
            grad_wrt (list[str], optional):
                A list of parameter names for which to keep gradients.  If None,
                gradients are taken with respect to all model parameters.
                Defaults to None.
        """
        super().__init__(model, task, grad_dim, dtype, device)
        self.model = model
        self.num_params = get_num_params(self.model)
        self.load_model_params(model)
        self.grad_wrt = grad_wrt
        self.logger = logging.getLogger("GradientComputer")

    def load_model_params(self, model) -> None:
        """Given a a torch.nn.Module model, inits/updates the (functional)
        weights and buffers. See https://pytorch.org/docs/stable/func.html
        for more details on :code:`torch.func`'s functional models.

        Args:
            model (torch.nn.Module):
                model to load

        """
        self.func_weights = dict(model.named_parameters())
        self.func_buffers = dict(model.named_buffers())

    def compute_per_sample_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Uses functorch's :code:`vmap` (see
        https://pytorch.org/functorch/stable/generated/functorch.vmap.html#functorch.vmap
        for more details) to vectorize the computations of per-sample gradients.

        Doesn't use :code:`batch_size`; only added to follow the abstract method
        signature.

        Args:
            batch (Iterable[Tensor]):
                batch of data

        Returns:
            dict[Tensor]:
                A dictionary where each key is a parameter name and the value is
                the gradient tensor for that parameter.

        """
        # taking the gradient wrt weights (second argument of get_output, hence argnums=1)
        grads_loss = torch.func.grad(
            self.modelout_fn.get_output, has_aux=False, argnums=1
        )

        # map over batch dimensions (hence 0 for each batch dimension, and None for model params)
        grads = torch.func.vmap(
            grads_loss,
            in_dims=(None, None, None, *([0] * len(batch))),
            randomness="different",
        )(self.model, self.func_weights, self.func_buffers, *batch)

        if self.grad_wrt is not None:
            for param_name in list(grads.keys()):
                if param_name not in self.grad_wrt:
                    del grads[param_name]
        return grads

    def compute_loss_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Computes the gradient of the loss with respect to the model output

        .. math::

            \\partial \\ell / \\partial \\text{(model output)}

        Note: For all applications we considered, we analytically derived the
        out-to-loss gradient, thus avoiding the need to do any backward passes
        (let alone per-sample grads). If for your application this is not feasible,
        you'll need to subclass this and modify this method to have a structure
        similar to the one of :meth:`FunctionalGradientComputer:.get_output`,
        i.e. something like:

        .. code-block:: python

            grad_out_to_loss = grad(self.model_out_to_loss_grad, ...)
            grads = vmap(grad_out_to_loss, ...)
            ...

        Args:
            batch (Iterable[Tensor]):
                batch of data

        Returns:
            Tensor:
                The gradient of the loss with respect to the model output.
        """
        return self.modelout_fn.get_out_to_loss_grad(
            self.model, self.func_weights, self.func_buffers, batch
        )


class TRAKer:
    """The main front-facing class for TRAK. See the `README
    <https://github.com/MadryLab/trak>`_ and `docs
    <https://trak.readthedocs.io/en/latest/>`_ for example usage.

    """

    def __init__(
        self,
        model: nn.Module,
        task: Union[AbstractModelOutput, str],
        train_set_size: int,
        save_dir: str = "./trak_results",
        load_from_save_dir: bool = True,
        device: Union[str, torch.device] = "cuda",
        gradient_computer: AbstractGradientComputer = FunctionalGradientComputer,
        projector: Optional[AbstractProjector] = None,
        saver: Optional[AbstractSaver] = None,
        score_computer: Optional[AbstractScoreComputer] = None,
        proj_dim: int = 2048,
        logging_level=logging.INFO,
        use_half_precision: bool = True,
        proj_max_batch_size: int = 32,
        projector_seed: int = 0,
        grad_wrt: Optional[Iterable[str]] = None,
        lambda_reg: float = 0.0,
    ) -> None:
        """

        Args:
            model (torch.nn.Module):
                model to use for TRAK
            task (Union[AbstractModelOutput, str]):
                Type of model that TRAK will be ran on. Accepts either one of
                the following strings: 1) :code:`image_classification` 2)
                :code:`text_classification` 3) :code:`clip` or an instance of
                some implementation of the abstract class
                :class:`.AbstractModelOutput`.
            train_set_size (int):
                Size of the train set that TRAK is featurizing
            save_dir (str, optional):
                Directory to save final TRAK scores, intermediate results, and
                metadata. Defaults to :code:'./trak_results'.
            load_from_save_dir (bool, optional):
                If True, the :class`.TRAKer` instance will attempt to load
                existing metadata from save_dir. May lead to I/O issues if
                multiple TRAKer instances ran in parallel have this flag set to
                True. See the SLURM tutorial for more details.
            device (Union[str, torch.device], optional):
                torch device on which to do computations. Defaults to 'cuda'.
            gradient_computer (AbstractGradientComputer, optional):
                Class to use to get per-example gradients. See
                :class:`.AbstractGradientComputer` for more details. Defaults to
                :class:`.FunctionalGradientComputer`.
            projector (Optional[AbstractProjector], optional):
                Either set :code:`proj_dim` and a :class:`.CudaProjector`
                Rademacher projector will be used or give a custom subclass of
                :class:`.AbstractProjector` class and leave :code:`proj_dim` as
                None. Defaults to None.
            saver (Optional[AbstractSaver], optional):
                Class to use for saving intermediate results and final TRAK
                scores to RAM/disk. If None, the :class:`.MmapSaver` will
                be used. Defaults to None.
            score_computer (Optional[AbstractScoreComputer], optional):
                Class to use for computing the final TRAK scores. If None, the
                :class:`.BasicScoreComputer` will be used. Defaults to None.
            proj_dim (int, optional):
                Dimension of the projected TRAK features. See Section 4.3 of
                `our paper <https://arxiv.org/abs/2303.14186>`_ for more
                details. Defaults to 2048.
            logging_level (int, optional):
                Logging level for TRAK loggers. Defaults to logging.INFO.
            use_half_precision (bool, optional):
                If True, TRAK will use half precision (float16) for all
                computations and arrays will be stored in float16. Otherwise, it
                will use float32. Defaults to True.
            proj_max_batch_size (int):
                Batch size used by fast_jl if the CudaProjector is used. Must be
                a multiple of 8. The maximum batch size is 32 for A100 GPUs, 16
                for V100 GPUs, 40 for H100 GPUs. Defaults to 32.
            projector_seed (int):
                Random seed used by the projector. Defaults to 0.
            grad_wrt (Optional[Iterable[str]], optional):
                If not None, the gradients will be computed only with respect to
                the parameters specified in this list. The list should contain
                the names of the parameters to compute gradients with respect to,
                as they appear in the model's state dictionary. If None,
                gradients are taken with respect to all model parameters.
                Defaults to None.
            lambda_reg (float):
                The :math:`\ell_2` (ridge) regularization penalty added to the
                :math:`XTX` term in score computers when computing the matrix
                inverse :math:`(XTX)^{-1}`. Defaults to 0.
        """

        self.model = model
        self.task = task
        self.train_set_size = train_set_size
        self.device = device
        self.dtype = torch.float16 if use_half_precision else torch.float32
        self.grad_wrt = grad_wrt
        self.lambda_reg = lambda_reg

        logging.basicConfig()
        self.logger = logging.getLogger("TRAK")
        self.logger.setLevel(logging_level) # logging.basicConfig(level=logging.DEBUG)

        self.num_params = get_num_params(self.model)
        if self.grad_wrt is not None:
            d = dict(self.model.named_parameters())
            self.num_params_for_grad = sum(
                [d[param_name].numel() for param_name in self.grad_wrt]
            )
        else:
            self.num_params_for_grad = self.num_params
        # inits self.projector
        self.proj_seed = projector_seed
        self.init_projector(
            projector=projector,
            proj_dim=proj_dim,
            proj_max_batch_size=proj_max_batch_size,
        )

        # normalize to make X^TX numerically stable
        # doing this instead of normalizing the projector matrix
        self.normalize_factor = torch.sqrt(
            torch.tensor(self.num_params_for_grad, dtype=torch.float32)
        )

        self.save_dir = Path(save_dir).resolve()
        self.load_from_save_dir = load_from_save_dir

        if type(self.task) is str:
            
            TASK_TO_MODELOUT = {
                "image_classification": ImageClassificationModelOutput,
                #"clip": CLIPModelOutput,
                #"text_classification": TextClassificationModelOutput,
                #"iterative_image_classification": IterativeImageClassificationModelOutput,
            }
            
            self.task = TASK_TO_MODELOUT[self.task]()

        self.gradient_computer = gradient_computer(
            model=self.model,
            task=self.task,
            grad_dim=self.num_params_for_grad,
            dtype=self.dtype,
            device=self.device,
            grad_wrt=self.grad_wrt,
        )

        if score_computer is None:
            score_computer = BasicScoreComputer
        self.score_computer = score_computer(
            dtype=self.dtype,
            device=self.device,
            logging_level=logging_level,
            lambda_reg=self.lambda_reg,
        )

        metadata = {
            "JL dimension": self.proj_dim,
            "JL matrix type": self.projector.proj_type,
            "train set size": self.train_set_size,
        }

        if saver is None:
            saver = MmapSaver
        self.saver = saver(
            save_dir=self.save_dir,
            metadata=metadata,
            train_set_size=self.train_set_size,
            proj_dim=self.proj_dim,
            load_from_save_dir=self.load_from_save_dir,
            logging_level=logging_level,
            use_half_precision=use_half_precision,
        )

        self.ckpt_loaded = "no ckpt loaded"

    def init_projector(
        self,
        projector: Optional[AbstractProjector],
        proj_dim: int,
        proj_max_batch_size: int,
    ) -> None:
        """Initialize the projector for a traker class

        Args:
            projector (Optional[AbstractProjector]):
                JL projector to use. If None, a CudaProjector will be used (if
                possible).
            proj_dim (int):
                Dimension of the projected gradients and TRAK features.
            proj_max_batch_size (int):
                Batch size used by fast_jl if the CudaProjector is used. Must be
                a multiple of 8. The maximum batch size is 32 for A100 GPUs, 16
                for V100 GPUs, 40 for H100 GPUs.
        """

        self.projector = projector
        if projector is not None:
            self.proj_dim = self.projector.proj_dim
            if self.proj_dim == 0:  # using NoOpProjector
                self.proj_dim = self.num_params_for_grad

        else:
            using_cuda_projector = False
            self.proj_dim = proj_dim
            if self.device == "cpu":
                self.logger.info("Using BasicProjector since device is CPU")
                projector = BasicProjector
                # Sampling from bernoulli distribution is not supported for
                # dtype float16 on CPU; playing it safe here by defaulting to
                # normal projection, rather than rademacher
                proj_type = ProjectionType.normal
                self.logger.info("Using Normal projection")
            else:
                try:
                    import fast_jl

                    test_gradient = torch.ones(1, self.num_params_for_grad).cuda()
                    num_sms = torch.cuda.get_device_properties(
                        "cuda"
                    ).multi_processor_count
                    fast_jl.project_rademacher_8(
                        test_gradient, self.proj_dim, 0, num_sms
                    )
                    projector = CudaProjector
                    using_cuda_projector = True

                except (ImportError, RuntimeError, AttributeError) as e:
                    self.logger.error(f"Could not use CudaProjector.\nReason: {str(e)}")
                    self.logger.error("Defaulting to BasicProjector.")
                    projector = BasicProjector
                proj_type = ProjectionType.rademacher

            if using_cuda_projector:
                max_chunk_size, param_chunk_sizes = get_parameter_chunk_sizes(
                    self.model, proj_max_batch_size
                )
                self.logger.debug(
                    (
                        f"the max chunk size is {max_chunk_size}, ",
                        "while the model has the following chunk sizes",
                        f"{param_chunk_sizes}.",
                    )
                )

                if (
                    len(param_chunk_sizes) > 1
                ):  # we have to use the ChunkedCudaProjector
                    self.logger.info(
                        (
                            f"Using ChunkedCudaProjector with"
                            f"{len(param_chunk_sizes)} chunks of sizes"
                            f"{param_chunk_sizes}."
                        )
                    )
                    rng = np.random.default_rng(self.proj_seed)
                    seeds = rng.integers(
                        low=0,
                        high=500,
                        size=len(param_chunk_sizes),
                    )
                    projector_per_chunk = [
                        projector(
                            grad_dim=chunk_size,
                            proj_dim=self.proj_dim,
                            seed=seeds[i],
                            proj_type=ProjectionType.rademacher,
                            max_batch_size=proj_max_batch_size,
                            dtype=self.dtype,
                            device=self.device,
                        )
                        for i, chunk_size in enumerate(param_chunk_sizes)
                    ]
                    self.projector = ChunkedCudaProjector(
                        projector_per_chunk,
                        max_chunk_size,
                        param_chunk_sizes,
                        proj_max_batch_size,
                        self.device,
                        self.dtype,
                    )
                    return  # do not initialize projector below

            self.logger.debug(
                f"Initializing projector with grad_dim {self.num_params_for_grad}"
            )
            self.projector = projector(
                grad_dim=self.num_params_for_grad,
                proj_dim=self.proj_dim,
                seed=self.proj_seed,
                proj_type=proj_type,
                max_batch_size=proj_max_batch_size,
                dtype=self.dtype,
                device=self.device,
            )
            self.logger.debug(f"Initialized projector with proj_dim {self.proj_dim}")

    def load_checkpoint(
        self,
        checkpoint: Iterable[Tensor],
        model_id: int,
        _allow_featurizing_already_registered=False,
    ) -> None:
        """Loads state dictionary for the given checkpoint; initializes arrays
        to store TRAK features for that checkpoint, tied to the model ID.

        Args:
            checkpoint (Iterable[Tensor]):
                state_dict to load
            model_id (int):
                a unique ID for a checkpoint
            _allow_featurizing_already_registered (bool, optional):
                Only use if you want to override the default behaviour that
                :code:`featurize` is forbidden on already registered model IDs.
                Defaults to None.

        """
        if self.saver.model_ids.get(model_id) is None:
            self.saver.register_model_id(
                model_id, _allow_featurizing_already_registered
            )
        else:
            self.saver.load_current_store(model_id)

        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.gradient_computer.load_model_params(self.model)

        self._last_ind = 0
        self.ckpt_loaded = model_id

    def featurize(
        self,
        batch: Iterable[Tensor],
        inds: Optional[Iterable[int]] = None,
        num_samples: Optional[int] = None,
    ) -> None:
        """Creates TRAK features for the given batch by computing the gradient
        of the model output function and projecting it. In the notation of the
        paper, for an input pair :math:`z=(x,y)`, model parameters
        :math:`\\theta`, and JL projection matrix :math:`P`, this method
        computes :math:`P^\\top \\nabla_\\theta f(z_i, \\theta)`.
        Additionally, this method computes the gradient of the out-to-loss
        function (in the notation of the paper, the :math:`Q` term in Section
        3.4).

        Either :code:`inds` or :code:`num_samples` must be specified. Using
        :code:`num_samples` will write sequentially into the internal store of
        the :func:`TRAKer`.

        Args:
            batch (Iterable[Tensor]):
                input batch
            inds (Optional[Iterable[int]], optional):
                Indices of the batch samples in the train set. Defaults to None.
            num_samples (Optional[int], optional):
                Number of samples in the batch. Defaults to None.

        """
        assert (
            self.ckpt_loaded == self.saver.current_model_id
        ), "Load a checkpoint using traker.load_checkpoint before featurizing"
        assert (inds is None) or (
            num_samples is None
        ), "Exactly one of num_samples and inds should be specified"
        assert (inds is not None) or (
            num_samples is not None
        ), "Exactly one of num_samples and inds should be specified"

        if num_samples is not None:
            inds = np.arange(self._last_ind, self._last_ind + num_samples)
            self._last_ind += num_samples
        else:
            num_samples = inds.reshape(-1).shape[0]

        # handle re-starting featurizing from a partially featurized model (some inds already featurized)
        _already_done = (self.saver.current_store["is_featurized"][inds] == 1).reshape(
            -1
        )
        inds = inds[~_already_done]
        if len(inds) == 0:
            self.logger.debug("All samples in batch already featurized.")
            return 0

        grads = self.gradient_computer.compute_per_sample_grad(batch=batch)
        grads = self.projector.project(grads, model_id=self.saver.current_model_id)
        grads /= self.normalize_factor
        self.saver.current_store["grads"][inds] = (
            grads.to(self.dtype).cpu().clone().detach()
        )

        loss_grads = self.gradient_computer.compute_loss_grad(batch)
        self.saver.current_store["out_to_loss"][inds] = (
            loss_grads.to(self.dtype).cpu().clone().detach()
        )

        self.saver.current_store["is_featurized"][inds] = 1
        self.saver.serialize_current_model_id_metadata()

    def finalize_features(
        self, model_ids: Iterable[int] = None, del_grads: bool = False
    ) -> None:
        """For a set of checkpoints :math:`C` (specified by model IDs), and
        gradients :math:`\\{ \\Phi_c \\}_{c\\in C}`, this method computes
        :math:`\\Phi_c (\\Phi_c^\\top\\Phi_c)^{-1}` for all :math:`c\\in C`
        and stores the results in the internal store of the :func:`TRAKer`
        class.

        Args:
            model_ids (Iterable[int], optional): A list of model IDs for which
                features should be finalized. If None, features are finalized
                for all model IDs in the :code:`save_dir` of the :class:`.TRAKer`
                class. Defaults to None.

        """

        # this method is memory-intensive, so we're freeing memory beforehand
        torch.cuda.empty_cache()
        self.projector.free_memory()

        if model_ids is None:
            model_ids = list(self.saver.model_ids.keys())

        self._last_ind = 0

        for model_id in tqdm(model_ids, desc="Finalizing features for all model IDs..", disable=self.logger.level > 20):
            if self.saver.model_ids.get(model_id) is None:
                raise ModelIDException(
                    f"Model ID {model_id} not registered, not ready for finalizing."
                )
            elif self.saver.model_ids[model_id]["is_featurized"] == 0:
                raise ModelIDException(
                    f"Model ID {model_id} not fully featurized, not ready for finalizing."
                )
            elif self.saver.model_ids[model_id]["is_finalized"] == 1:
                self.logger.warning(
                    f"Model ID {model_id} already finalized, skipping .finalize_features for it."
                )
                continue

            self.saver.load_current_store(model_id)

            g = torch.as_tensor(self.saver.current_store["grads"], device=self.device)
            xtx = self.score_computer.get_xtx(g)

            features = self.score_computer.get_x_xtx_inv(g, xtx)
            self.saver.current_store["features"][:] = features.to(self.dtype).cpu()
            if del_grads:
                self.saver.del_grads(model_id)

            self.saver.model_ids[self.saver.current_model_id]["is_finalized"] = 1
            self.saver.serialize_current_model_id_metadata()

    def start_scoring_checkpoint(
        self,
        exp_name: str,
        checkpoint: Iterable[Tensor],
        model_id: int,
        num_targets: int,
    ) -> None:
        """This method prepares the internal store of the :class:`.TRAKer` class
        to start computing scores for a set of targets.

        Args:
            exp_name (str):
                Experiment name. Each experiment should have a unique name, and
                it corresponds to a set of targets being scored. The experiment
                name is used as the name for saving the target features, as well
                as scores produced by this method in the :code:`save_dir` of the
                :class:`.TRAKer` class.
            checkpoint (Iterable[Tensor]):
                model checkpoint (state dict)
            model_id (int):
                a unique ID for a checkpoint
            num_targets (int):
                number of targets to score

        """
        self.saver.init_experiment(exp_name, num_targets, model_id)

        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.gradient_computer.load_model_params(self.model)

        # TODO: make this exp_name-dependent
        # e.g. make it a value in self.saver.experiments[exp_name]
        self._last_ind_target = 0

    def score(
        self,
        batch: Iterable[Tensor],
        inds: Optional[Iterable[int]] = None,
        num_samples: Optional[int] = None,
    ) -> None:
        """This method computes the (intermediate per-checkpoint) TRAK scores
        for a batch of targets and stores them in the internal store of the
        :class:`.TRAKer` class.

        Either :code:`inds` or :code:`num_samples` must be specified. Using
        :code:`num_samples` will write sequentially into the internal store of
        the :class:`.TRAKer`.

        Args:
            batch (Iterable[Tensor]):
                input batch
            inds (Optional[Iterable[int]], optional):
                Indices of the batch samples in the train set. Defaults to None.
            num_samples (Optional[int], optional):
                Number of samples in the batch. Defaults to None.

        """
        assert (inds is None) or (
            num_samples is None
        ), "Exactly one of num_samples and inds should be specified"
        assert (inds is not None) or (
            num_samples is not None
        ), "Exactly one of num_samples and inds should be specified"

        if self.saver.model_ids[self.saver.current_model_id]["is_finalized"] == 0:
            self.logger.error(
                f"Model ID {self.saver.current_model_id} not finalized, cannot score"
            )
            return None

        if num_samples is not None:
            inds = np.arange(self._last_ind_target, self._last_ind_target + num_samples)
            self._last_ind_target += num_samples
        else:
            num_samples = inds.reshape(-1).shape[0]

        grads = self.gradient_computer.compute_per_sample_grad(batch=batch)

        grads = self.projector.project(grads, model_id=self.saver.current_model_id)
        grads /= self.normalize_factor

        exp_name = self.saver.current_experiment_name
        self.saver.current_store[f"{exp_name}_grads"][inds] = (
            grads.to(self.dtype).cpu().clone().detach()
        )

    def finalize_scores(
        self,
        exp_name: str,
        model_ids: Iterable[int] = None,
        allow_skip: bool = False,
    ) -> Tensor:
        """This method computes the final TRAK scores for the given targets,
        train samples, and model checkpoints (specified by model IDs).

        Args:
            exp_name (str):
                Experiment name. Each experiment should have a unique name, and
                it corresponds to a set of targets being scored. The experiment
                name is used as the name for saving the target features, as well
                as scores produced by this method in the :code:`save_dir` of the
                :class:`.TRAKer` class.
            model_ids (Iterable[int], optional):
                A list of model IDs for which
                scores should be finalized. If None, scores are computed
                for all model IDs in the :code:`save_dir` of the :class:`.TRAKer`
                class. Defaults to None.
            allow_skip (bool, optional):
                If True, raises only a warning, instead of an error, when target
                gradients are not computed for a given model ID. Defaults to
                False.

        Returns:
            Tensor: TRAK scores

        """
        # reset counter for inds used for scoring
        self._last_ind_target = 0

        if model_ids is None:
            model_ids = self.saver.model_ids
        else:
            model_ids = {
                model_id: self.saver.model_ids[model_id] for model_id in model_ids
            }
        assert len(model_ids) > 0, "No model IDs to finalize scores for"

        if self.saver.experiments.get(exp_name) is None:
            raise ValueError(
                f"Experiment {exp_name} does not exist. Create it\n\
                              and compute scores first before finalizing."
            )

        num_targets = self.saver.experiments[exp_name]["num_targets"]
        _completed = [False] * len(model_ids)

        self.saver.load_current_store(list(model_ids.keys())[0], exp_name, num_targets)
        _scores_mmap = self.saver.current_store[f"{exp_name}_scores"]
        _scores_on_cpu = torch.zeros(*_scores_mmap.shape, device="cpu")
        if self.device != "cpu":
            _scores_on_cpu.pin_memory()

        _avg_out_to_losses = np.zeros(
            (self.saver.train_set_size, 1),
            dtype=np.float16 if self.dtype == torch.float16 else np.float32,
        )

        for j, model_id in enumerate(
            tqdm(model_ids, desc="Finalizing scores for all model IDs..", disable=self.logger.level > 20)
        ):
            self.saver.load_current_store(model_id)
            try:
                self.saver.load_current_store(model_id, exp_name, num_targets)
            except OSError as e:
                if allow_skip:
                    self.logger.warning(
                        f"Could not read target gradients for model ID {model_id}. Skipping."
                    )
                    continue
                else:
                    raise e

            if self.saver.model_ids[self.saver.current_model_id]["is_finalized"] == 0:
                self.logger.warning(
                    f"Model ID {self.saver.current_model_id} not finalized, cannot score"
                )
                continue

            g = torch.as_tensor(self.saver.current_store["features"], device=self.device)
            g_target = torch.as_tensor(
                self.saver.current_store[f"{exp_name}_grads"], device=self.device
            )

            self.score_computer.get_scores(g, g_target, accumulator=_scores_on_cpu)
            # .cpu().detach().numpy()

            _avg_out_to_losses += self.saver.current_store["out_to_loss"]
            _completed[j] = True

        _num_models_used = float(sum(_completed))

        # only write to mmap (on disk) once at the end
        _scores_mmap[:] = (_scores_on_cpu.numpy() / _num_models_used) * (
            _avg_out_to_losses / _num_models_used
        )

        self.logger.debug(f"Scores dtype is {_scores_mmap.dtype}")
        self.saver.save_scores(exp_name)
        self.scores = _scores_mmap

        return self.scores
    

def trak(model, checkpoints_dir, train_dataloader, targets_loader, train_set_size, num_samples, max_models = 15):

    traker = TRAKer(model=model, 
                    task='image_classification', 
                    train_set_size=train_set_size, 
                    save_dir=os.path.join(checkpoints_dir, "trak"),
                    logging_level= 40)

    ckpt_files = sorted(list(Path(checkpoints_dir).rglob('*.pt')))
    checkpoints = [torch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]
    # max_models = min(max_models, len(checkpoints))
    relevant_ids = []

    for model_id, checkpoint in enumerate(checkpoints):
        if model_id < len(checkpoints) - max_models:
            continue
        relevant_ids.append(model_id)
        traker.load_checkpoint(checkpoint, model_id=model_id)
        for batch in train_dataloader:
            batch = [x.cuda() for x in batch]
            if len(batch[-1].shape) > 1:
                batch[-1] = torch.argmax(batch[-1], dim=1)
            # batch should be a tuple of inputs and labels
            traker.featurize(batch=batch, num_samples=batch[0].shape[0])
        traker.finalize_features()

    for model_id, checkpoint in enumerate(checkpoints):
        if model_id < len(checkpoints) - max_models:
            continue
        relevant_ids.append(model_id)
        traker.start_scoring_checkpoint(checkpoint=checkpoint,
                                        model_id=model_id,
                                        exp_name='test',
                                        num_targets=num_samples)
        for batch in targets_loader:
            batch = [x.cuda() for x in batch]
            if len(batch[-1].shape) > 1:
                batch[-1] = torch.argmax(batch[-1], dim=1)
            traker.score(batch=batch, num_samples=batch[0].shape[0])

    scores = traker.finalize_scores(exp_name='test', model_ids=relevant_ids)

    return scores

def visualize(scores, train_dataset, test_dataset, save_path = None, inds = None):
    if inds is None:
        inds = range(10)
    figs = []

    scores -= scores.min()
    scores /= scores.max()

    for i in inds:

        fig, axs = plt.subplots(nrows=2, ncols=7, figsize=(15, 3))
        fig.suptitle('Top/bot scoring TRAK images from the train set')
        
        axs[0, 0].imshow(test_dataset[i][0].permute(2, 1, 0))
        axs[0, 0].axis('off')
        axs[0, 0].set_title('Target image')
        axs[0, 1].axis('off')
        axs[1, 0].imshow(test_dataset[i][0].permute(2, 1, 0))
        axs[1, 0].axis('off')
        axs[1, 0].set_title('Target image')
        axs[1, 1].axis('off')
        
        indices = scores[:, i].argsort()
        top_trak_scorers_indices = indices[-5:][::-1]
        top_trak_scorers_values = scores[:, i][top_trak_scorers_indices]
        bot_trak_scorers_indices = indices[:5][::-1]
        bot_trak_scorers_values = scores[:, i][bot_trak_scorers_indices]
        for ii, train_im_ind in enumerate(top_trak_scorers_indices):
            axs[0, ii + 2].imshow(train_dataset[train_im_ind][0].permute(2, 1, 0))
            axs[0, ii + 2].axis('off')
            axs[0, ii + 2].set_title(f"{round(top_trak_scorers_values[ii], 5)}")

            axs[1, ii + 2].imshow(train_dataset[train_im_ind][0].permute(2, 1, 0))
            axs[1, ii + 2].axis('off')
            axs[1, ii + 2].set_title(f"{round(bot_trak_scorers_values[ii], 5)}")

        figs.append(fig)
        if save_path is not None:
            plt.savefig(os.path.join(save_path, f"viz_ind_{i}.png", ))
    
    return figs

def trak_onebatch(model, checkpoints_dir, train_dataloader, target_batch, train_set_size, num_samples, max_models = 15, save_dir_end = None):
    save_dir = os.path.join(checkpoints_dir, "trak")
    if save_dir_end is not None:
        save_dir = os.path.join(save_dir, save_dir_end)
    traker = TRAKer(model=model, task='image_classification', train_set_size=train_set_size, save_dir=save_dir, logging_level=40) # "CRITICAL"

    ckpt_files = sorted(list(Path(checkpoints_dir).rglob('*.pt')))
    checkpoints = [torch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]
    max_models = min(max_models, len(checkpoints))

    for model_id, checkpoint in enumerate(checkpoints[:max_models]):
        traker.load_checkpoint(checkpoint, model_id=model_id)
        for batch in train_dataloader:
            batch = [x.cuda() for x in batch]
            if len(batch[-1].shape) > 1:
                batch[-1] = torch.argmax(batch[-1], dim=1)
            # batch should be a tuple of inputs and labels
            traker.featurize(batch=batch, num_samples=batch[0].shape[0])
        traker.finalize_features()

    for model_id, checkpoint in enumerate(checkpoints[:max_models]):
        traker.start_scoring_checkpoint(checkpoint=checkpoint,
                                        model_id=model_id,
                                        exp_name='test',
                                        num_targets=num_samples)
        target_batch = [x.cuda() for x in target_batch]
        if len(target_batch[-1].shape) > 1:
            target_batch[-1] = torch.argmax(target_batch[-1], dim=1)
        traker.score(batch=target_batch, num_samples=target_batch[0].shape[0])

    scores = traker.finalize_scores(exp_name='test', model_ids=range(max_models))

    return scores
