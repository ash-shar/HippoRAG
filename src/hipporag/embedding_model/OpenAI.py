from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel
from openai import OpenAI
from openai import AzureOpenAI
import os

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)

class OpenAIEmbeddingModel(BaseEmbeddingModel):

	def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
		super().__init__(global_config=global_config)

		if embedding_model_name is not None:
			self.embedding_model_name = embedding_model_name
			logger.debug(
				f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

		self._init_embedding_config()

		# Initializing the embedding model
		logger.debug(
			f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")
		
		api_key = os.getenv("OPENAI_API_KEY")
		
		self.client = AzureOpenAI(
								# this is the AOAI-east-us endpoint
								azure_endpoint = os.getenv("AOAI_EAST_US_ENDPOINT"),
								api_key = api_key,
								api_version="2024-10-21"
							)		

	def _init_embedding_config(self) -> None:
		"""
		Extract embedding model-specific parameters to init the EmbeddingConfig.

		Returns:
			None
		"""

		config_dict = {
			"embedding_model_name": self.embedding_model_name,
			"norm": self.global_config.embedding_return_as_normalized,
			# "max_seq_length": self.global_config.embedding_max_seq_len,
			"model_init_params": {
				# "model_name_or_path": self.embedding_model_name2mode_name_or_path[self.embedding_model_name],
				"pretrained_model_name_or_path": self.embedding_model_name,
				"trust_remote_code": True,
				# "torch_dtype": "auto",
				'device_map': "auto",  # added this line to use multiple GPUs
				# **kwargs
			},
			"encode_params": {
				"max_length": self.global_config.embedding_max_seq_len,  # 32768 from official example,
				"instruction": "",
				"batch_size": self.global_config.embedding_batch_size,
				"num_workers": 32
			},
		}

		self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
		logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

	def batch_encode(self, texts: List[str], **kwargs) -> None:
		if isinstance(texts, str): texts = [texts]

		params = deepcopy(self.embedding_config.encode_params)
		if kwargs: params.update(kwargs)

		if "instruction" in kwargs:
			if kwargs["instruction"] != '':
				params["instruction"] = f"Instruct: {kwargs['instruction']}\nQuery: "
			# del params["instruction"]

		logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")
		texts = [t.replace("\n", " ") for t in texts]

		assert all(len(t) > 0 for t in texts)  # embeddings.create throws Exception on empty string
		# assert all(len(tokenizer.encode(x)) <= 8192 for x in texts)

		batch_size = 800

		batches = np.array_split(texts, 1 + (len(texts) // batch_size))

		results = []

		for batch in batches:
			results.extend(np.array([e.embedding for e in self.client.embeddings.create(input=batch, model=self.embedding_model_name).data]))
		
		results = np.array(results)
		# results = np.array([v.embedding for v in response.data])

		if isinstance(results, torch.Tensor):
			results = results.cpu()
			results = results.numpy()
		if self.embedding_config.norm:
			results = (results.T / np.linalg.norm(results, axis=1)).T

		return results