from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments,AutoModelWithLMHead
from transformers import pipeline

#Check the len of data files
def read_data():
    file1 = open('/Users/ahadmushir/Documents/semester_c/Thesis/conceptual_solution/data/template.txt', 'r')
    train_data = file1.readlines()
    file2 = open('/Users/ahadmushir/Documents/semester_c/Thesis/conceptual_solution/data/template_test.txt', 'r')
    test_data = file2.readlines()
    return (train_data, test_data)


def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator

if __name__ == "__main__":
    dataset = read_data()
    tr = dataset[0]
    te = dataset[1]
    print(len(dataset[0]), len(dataset[1]))

    # tokenizer = AutoTokenizer.from_pretrained("anonymous-german-nlp/german-gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained('SIC98/GPT2-python-code-generator')

    train_path = '/Users/ahadmushir/Documents/semester_c/Thesis/conceptual_solution/data/template.txt'
    test_path = '/Users/ahadmushir/Documents/semester_c/Thesis/conceptual_solution/data/template_test.txt'

    train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)
    model = GPT2LMHeadModel.from_pretrained('SIC98/GPT2-python-code-generator')

    training_args = TrainingArguments(
        output_dir="./gpt2-gerchef",  # The output directory
        overwrite_output_dir=True,  # overwrite the content of the output directory
        num_train_epochs=3,  # number of training epochs
        per_device_train_batch_size=32,  # batch size for training
        per_device_eval_batch_size=64,  # batch size for evaluation
        eval_steps=400,  # Number of update steps between two evaluations.
        save_steps=800,  # after # steps model is saved
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # trainer.train()
    # trainer.save_model()

    try:
        code_gen = pipeline('text-generation', model='./gpt2-gerchef', tokenizer="SIC98/GPT2-python-code-generator")
    except Exception as e:
        print(e)

    print(code_gen("<annotation_start>define a function name addition which adds two parameters<annotation_end>"))

    # sequence = """
    # <annotation_start>define a function name addition which adds two parameters<annotation_end>
    # """
    # inputs = tokenizer.encode(sequence, return_tensors='pt')
    # outputs = model.generate(inputs, max_length=1024, do_sample=True, temperature=0.5, top_p=1.0)
    # text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # print(text)

    print("done")

