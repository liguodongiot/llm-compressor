"""
Code for downloading models from the Neural Magic Repository
"""

from typing import List, Union
import os
import json
import requests
from requests.exceptions import HTTPError
import hashlib
import copy
import tarfile

from neuralmagicML.utils.frameworks import (
    ONNX_FRAMEWORK,
    PYTORCH_FRAMEWORK,
    TENSORFLOW_FRAMEWORK,
)
from neuralmagicML.utils import clean_path, download_file

__all__ = [
    "models_sign_url",
    "models_download_file",
    "RepoModel",
    "filter_model",
    "available_models",
]


_REPO_BASE_URI = "https://models.neuralmagic.com"
_SIGNATURE_URI = "https://api.neuralmagic.com/models/sign-url"
_NM_SIGNING_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhY2Nlc3MiOiJhbGwiLCJleHAiOjE5NDQ0OTUwNDMsI"
    "nVzZXJJZCI6IjEyMzQ1NiIsImFwcElkIjoiNzg5MDEyIn0.G4LlNb7DOqn1oxkKELTtvgR_laeYDM22YpJ"
    "SgmP0Zkl82g5nqiGpHg9hegABPiUPCaTpqFaKRviDmaVZO6OeiiB3aXTlfYTYL26X0GJfcTK1l3O7PouxY"
    "rlN35l31rgYtTk4qyIPH1o6JmNRwuxIe1N_OZ2Vi7XQk46HmyS1GuhOOKC1UG3IoPPT9bxIF5uepZjQYN3"
    "bNQ-QmRi_ldWPe1Pl-kXD59H31PkjCNzhnVqt_FQI27lLgjDdhhbqHMzrUrOqXnR3WsJTkIajrWDFbMWKI"
    "9OJuAcaMATkWoO2ssUH7ADMYAR5B3bWbZKlXPZjWIjbNhIJANJNVg_F5i6OH4xVvqdLWnzNkmBShKwOCnF"
    "GTpRjCEgISHgfwl9CyTT5GT8XJA6VFaNFmHyK-TLWSCwir70eRRwUkApBva0KA51VAIv5YW2zmpscyWGHs"
    "DS8OTi0Ry7PZSbssJFv7zke7xY95NH54aMP1BbpIHwDPz1UNYtqp1WLZvE_TvGW3JsYTasZiOhaNYrlc1h"
    "ssaxZM2rMccaV5I8XVyE5jIn1jCgkw-7ETTklN-440FLeCsEbZTm5g3Zmw0nE35AfR1uawpCGa0hx2lH0Q"
    "59JTSVgQRJmOiYw0XpINsL3OznzyaOxJSIfzQbgq63nNCaqcW9uM1pOyoNPYPHvlTqkaZMh3pE"
)


def models_sign_url(path: str) -> str:
    """
    Create a signed url for use in models.neuralmagic.com repo

    :param path: the path of the desired url to sign
    :return: the signed url that can be used to get the path
    """
    headers = {"nm-token-header": _NM_SIGNING_TOKEN}
    model_req_data = {"body": {"redirect_path": path}}
    model_req_data = json.dumps(model_req_data)
    request = requests.post(url=_SIGNATURE_URI, data=model_req_data, headers=headers)
    request.raise_for_status()
    model_res_data = request.json()

    return model_res_data["signed_url"]


def models_download_file(path: str, overwrite: bool, save_dir: str = None) -> str:
    """
    Download the given file from models.neuralmagic.com repo

    :param path: the path for the file to download
    :param overwrite: True to overwrite the file if it exists, False otherwise
    :param save_dir: The directory to save the model files to
        instead of the default cache dir
    :return: the local path to the downloaded file
    """
    file_dir = "/".join(path.split("/")[:-1])
    file_name = path.split("/")[-1]
    url = models_sign_url(path)

    if not save_dir:
        save_name = hashlib.md5(file_dir.encode("utf-8")).hexdigest()
        save_path = os.getenv("NM_ML_MODELS_PATH", "")

        if not save_path:
            save_path = os.path.join("~", ".cache", "nm_models")

        save_dir = os.path.join(save_path, save_name)

    save_dir = clean_path(save_dir)
    save_file = os.path.join(save_dir, file_name)

    if overwrite and os.path.exists(save_file):
        try:
            os.remove(save_file)
        except OSError as err:
            print(
                "warning, error encountered when removing older "
                "cache_file at {}: {}".format(save_file, err)
            )

    if not os.path.exists(save_file) or overwrite:
        download_file(
            url, save_file, overwrite, progress_title="downloading {}".format(path)
        )

    return save_file


class RepoModel(object):
    """
    Class representing a model stored in the Neural Magic Repo.
    Offers convenience functions to download model files and information

    :param domain: the domain the model belongs to; ex: cv, nlp, etc
    :param sub_domain: the sub domain the model belongs to;
        ex: classification, detection, etc
    :param architecture: the architecture the model belongs to;
        ex: resnet, mobilenet, etc
    :param sub_architecture: the sub architecture the model belongs to;
        ex: 50, 101, etc
    :param dataset: the dataset used to train the model;
        ex: imagenet, cifar, etc
    :param framework: the framework used to train the model;
        ex: tensorflow, pytorch, keras, onnx, etc
    :param desc: the description of the model;
        ex: base, recal, recal-perf
    :param arch_display: human readable display for the model
    :param summary: human readable details about the model
    """

    def __init__(
        self,
        domain: str,
        sub_domain: str,
        architecture: str,
        sub_architecture: str,
        dataset: str,
        framework: str,
        desc: str,
        arch_display: Union[str, None] = None,
        summary: Union[str, None] = None,
    ):
        self.domain = domain
        self.sub_domain = sub_domain
        self.architecture = architecture
        self.sub_architecture = sub_architecture
        self.dataset = dataset
        self.framework = framework
        self.desc = desc
        self.arch_display = arch_display
        self.summary = summary

    def __repr__(self):
        return '{}(root_path={}, arch_display={}, summary="{}")'.format(
            self.__class__.__name__, self.root_path, self.arch_display, self.summary,
        )

    @property
    def registry_key(self) -> str:
        """
        :return: a key that can be used to retrieve a model from a ModelRegistry
            implementation in the neuralmagicML code
        """
        architecture = self.architecture.replace("-", "")

        if not self.sub_architecture or self.sub_architecture == "none":
            return architecture

        return "{}_{}".format(architecture, self.sub_architecture)

    @property
    def domain_display(self):
        """
        :return: human friendly display for the domain and subdomain
        """
        return "{} {}".format(self.domain.upper(), self.sub_domain.capitalize())

    @property
    def architecture_display(self):
        """
        :return: human friendly display for the architecture
        """
        if self.arch_display:
            return self.arch_display

        if self.sub_architecture == "none":
            return self.architecture

        return "{} {}".format(self.architecture, self.sub_architecture)

    @property
    def root_path(self) -> str:
        """
        :return: the combined root path the model will be found at in
            models.neuralmagic.com
        """

        return "{}-{}/{}/{}/{}/{}/{}".format(
            self.domain,
            self.sub_domain,
            self.architecture,
            self.sub_architecture,
            self.dataset,
            self.framework,
            self.desc,
        )

    @property
    def onnx_file_path(self) -> str:
        """
        :return: path the onnx file for the model will be found at in
            models.neuralmagic.com
        """
        return "{}/{}".format(self.root_path, "model.onnx")

    @property
    def data_file_paths(self) -> List[str]:
        """
        :return: path the data files for the model will be found at in
            models.neuralmagic.com
        """
        return [
            "{}/{}".format(self.root_path, "_sample-inputs.tar"),
            "{}/{}".format(self.root_path, "_sample-outputs.tar"),
            "{}/{}".format(self.root_path, "_sample-labels.tar"),
        ]

    @property
    def framework_file_paths(self) -> List[str]:
        """
        :return: path the framework specific file for the model will be found at in
            models.neuralmagic.com
        """
        if self.framework == ONNX_FRAMEWORK:
            return []

        if self.framework == PYTORCH_FRAMEWORK:
            return ["{}/{}".format(self.root_path, "model.pth")]

        if self.framework == TENSORFLOW_FRAMEWORK:
            return [
                "{}/{}".format(self.root_path, "model.pb"),
                "{}/{}".format(self.root_path, "model.meta"),
                "{}/{}".format(self.root_path, "model.index"),
                "{}/{}".format(self.root_path, "model.data-00000-of-00001"),
            ]

        raise ValueError("unsupported framework given of {}".format(self.framework))

    def download_onnx_file(self, overwrite: bool = False, save_dir: str = None) -> str:
        """
        :param overwrite: True to overwrite any previous downloads,
            False to not redownload if it exists (default)
        :param save_dir: The directory to save the model files to
            instead of the default cache dir
        :return: the path to the local, downloaded file
        """
        return models_download_file(self.onnx_file_path, overwrite, save_dir)

    def download_framework_files(
        self, overwrite: bool = False, save_dir: str = None
    ) -> List[str]:
        """
        :param overwrite: True to overwrite any previous downloads,
            False to not redownload if it exists (default)
        :param save_dir: The directory to save the model files to
            instead of the default cache dir
        :return: the path to the local, downloaded file
        """
        paths = []

        for framework_path in self.framework_file_paths:
            paths.append(models_download_file(framework_path, overwrite, save_dir))

        return paths

    def download_data_files(
        self, overwrite: bool = False, save_dir: str = None
    ) -> List[str]:
        """
        :param overwrite: True to overwrite any previous downloads,
            False to not redownload if it exists (default)
        :param save_dir: The directory to save the model files to
            instead of the default cache dir
        :return: the path to the local, downloaded file
        """
        paths = []

        for data_path in self.data_file_paths:
            try:
                paths.append(models_download_file(data_path, overwrite, save_dir))
                with tarfile.open(paths[-1]) as tar:
                    tar.extractall(save_dir)
            except HTTPError:
                print("Could not download {data_path}".format(data_path=data_path))
        return paths


_AVAILABLE_MODELS = None


def filter_model(
    model: RepoModel,
    domains: Union[List[str], None],
    sub_domains: Union[List[str], None],
    architectures: Union[List[str], None],
    sub_architectures: Union[List[str], None],
    datasets: Union[List[str], None],
    frameworks: Union[List[str], None],
    descs: Union[List[str], None],
) -> bool:
    """
    :param model: the model to decide if it should be filtered or not
    :param domains: the domains the model must belong to; ex: cv, nlp, etc
    :param sub_domains: the sub domains the model must belong to;
        ex: classification, detection, etc
    :param architectures: the architecture the model must belong to;
        ex: resnet, mobilenet, etc
    :param sub_architectures: the sub architectures the model must belong to;
        ex: 50, 101, etc
    :param datasets: the datasets the model must have been trained on; ex:
        imagenet, cifar, etc
    :param frameworks: the frameworks the model must have been trained in;
        ex: tensorflow, pytorch, etc
    :param descs: the descriptions the model must have;
        ex: base, recal, recal-perf
    :return: True if the model meets all filtering criteria, False otherwise
    """
    if domains and model.domain not in domains:
        return True

    if sub_domains and model.sub_domain not in sub_domains:
        return True

    if architectures and model.architecture not in architectures:
        return True

    if sub_architectures and model.sub_architecture not in sub_architectures:
        return True

    if datasets and model.dataset not in datasets:
        return True

    if frameworks and model.framework not in frameworks:
        return True

    if descs and model.desc not in descs:
        return True

    return False


def available_models(
    domains: Union[List[str], None] = None,
    sub_domains: Union[List[str], None] = None,
    architectures: Union[List[str], None] = None,
    sub_architectures: Union[List[str], None] = None,
    datasets: Union[List[str], None] = None,
    frameworks: Union[List[str], None] = None,
    descs: Union[List[str], None] = None,
) -> List[RepoModel]:
    """
    Get a list of the available models in the Neural Magic Model Repo.
    Additionally, the models can be filtered to only desired model criteria.

    :param domains: the domains the model must belong to; ex: cv, nlp, etc
    :param sub_domains: the sub domains the model must belong to;
        ex: classification, detection, etc
    :param architectures: the architecture the model must belong to;
        ex: resnet, mobilenet, etc
    :param sub_architectures: the sub architectures the model must belong to;
        ex: 50, 101, etc
    :param datasets: the datasets the model must have been trained on;
        ex: imagenet, cifar, etc
    :param frameworks: the frameworks the model must have been trained in;
        ex: tensorflow, pytorch, etc
    :param descs: the descriptions the model must have;
        ex: base, recal, recal-perf
    :return: the available (and potentially filtered) models
    """
    global _AVAILABLE_MODELS

    if _AVAILABLE_MODELS is None:
        file_path = models_download_file("available.json", overwrite=True)

        with open(file_path, "r") as file:
            models_dicts = json.load(file)["models"]

        _AVAILABLE_MODELS = []
        for mod in models_dicts:
            _AVAILABLE_MODELS.append(RepoModel(**mod))

    models = []

    for mod in _AVAILABLE_MODELS:
        if not filter_model(
            mod,
            domains,
            sub_domains,
            architectures,
            sub_architectures,
            datasets,
            frameworks,
            descs,
        ):
            models.append(copy.copy(mod))

    return models
