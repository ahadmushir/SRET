import logging

from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = LanguageModelingArgs()

model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 3
model_args.dataset_type = "simple"
model_args.special_tokens	= ['<annotation_start>', '<annotation_end>', '<code_start>', '<code_end>']
model_args.mlm = False  # mlm must be False for CLM

train_file = "template1_train.txt"
test_file = "template1_test.txt"

model = LanguageModelingModel(
    "gpt2", "gpt2-medium", args=model_args, use_cuda=True
)

# Train the model
model.train_model(train_file, eval_file=test_file)

# Evaluate the model
result = model.eval_model(test_file)
