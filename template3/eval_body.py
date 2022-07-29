from transformers import pipeline
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer, GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
)

model1 = GPT2LMHeadModel.from_pretrained("./outputs", pad_token_id=tokenizer.eos_token_id)

code_gen = pipeline('text-generation', model=model1, tokenizer=tokenizer)

#print(code_gen("<annotation_start_b>for every element i in the list xyzDemo, print the value i and assign the value i with demoxyz.<annotation_end_b>", max_length=200))

print(code_gen("<annotation_start_b>for every element i from range 1 to 10, multiply every element by 2.<annotation_end_b>", max_length=200))

'''
input_ids = tokenizer.encode('define the function _create_cache with backend and dictionary pair of elements kwargs as arguments', return_tensors='pt')

generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
'''


#code_gen = pipeline('text-generation', model="./saved_model_1", tokenizer=tokenizer)

#print(code_gen("<annotation_start_m>  define the function _create_cache with backend and dictionary pair of elements kwargs as arguments <annotation_end_m>", max_length=100))

'''
print(code_gen("<annotation_start_b>  multiply x and y and increment the result by 1  <annotation_end_b>", max_length=300))
'''

'''
text_generation = pipeline("text-generation", tokenizer=tokenizer,model="/home/mushir/saved_model_template2_3")
prefix_text = "<annotation_start_m>"
#prefix_text = tokenizer.encode(prefix_text)

generated_text= text_generation(prefix_text, max_length=50, do_sample=False)[0]
t = tokenizer.decode(generated_text['generated_text'])
'''
#print(generated_text['generated_text'].encode('utf-8'))

