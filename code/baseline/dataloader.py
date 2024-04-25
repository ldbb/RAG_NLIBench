import os
import json
import nltk

def train_data(input_dir):
    '''
    This function generates a training dataset from Super Natural Instruction datasets.
    '''

    data_list = []
    output_data = []
    
    # Walk through the specified directory to find .json files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    
                # Loop through the keys 'Positive Examples' and 'Instances'
                for key in ['Positive Examples', 'Instances']:
                    for item in json_data[key][:40]:

                        # Check if the 'output' is a list and take the first element
                        if isinstance(item['output'], list):
                            item['output'] = item['output'][0]
                            

                            # Create a data dictionary containing 'instruction', 'input', 'output', and 'category'
                            data = {
                                "instruction": json_data['Definition'][0],
                                "input": item['input'],
                                "output": item['output'],
                                "category": json_data['Categories'][0]
                            }

                            # Tokenize the 'instruction' and 'input' and count the number of tokens, if the number of tokens is less than 256, append the data to the data_list and output_data, lora cutoff=256 token, so we choose those data less than 256 token
                            tokens = nltk.word_tokenize(data['instruction'] + data['input'])
                            num_token = len(tokens)

                            if num_token < 256:
                                data_list.append(data)
                                output_data.append(data)

                            # If the length of data_list is 40, reset the data_list and break the loop
                            if len(data_list) == 40:
                                data_list = []
                                break  
    
    
    # Write the output_data to a .json file
    with open(os.path.join(input_dir, 'train.json'), 'w', encoding='utf-8') as file:
        json.dump(output_data, file, indent=4, ensure_ascii=False)
    
    # Print the length of output_data
    print(len(output_data))


def test_data(input_dir):
    '''
    This function generates a test dataset from Super Natural Instruction datasets.
    '''

    data_list = []
    output_data = []
    
    # Walk through the specified directory to find .json files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    
                # Loop through the key 'Instances'
                for key in ['Instances']:
                    for item in json_data[key][40:]:

                        # Check if the 'output' is a list and take the first element
                        if isinstance(item['output'], list):
                            item['output'] = item['output'][0]
                            

                            # Create a data dictionary containing 'instruction', 'input', 'output', and 'category'
                            data = {
                                "instruction": json_data['Definition'][0],
                                "input": item['input'],
                                "output": item['output'],
                                "category": json_data['Categories'][0]
                            }

                            # Tokenize the 'instruction' and 'input' and count the number of tokens, if the number of tokens is less than 256, append the data to the data_list and output_data, lora cutoff=256 token, so we choose those data less than 256 token
                            tokens = nltk.word_tokenize(data['instruction'] + data['input'])
                            num_token = len(tokens)

                            if num_token < 256:
                                data_list.append(data)
                                output_data.append(data)

                            # If the length of data_list is 40, reset the data_list and break the loop
                            if len(data_list) == 40:
                                data_list = []
                                break  
    
    # Write the output_data to a .json file
    with open(os.path.join(input_dir, 'test.json'), 'w', encoding='utf-8') as file:
        json.dump(output_data, file, indent=4, ensure_ascii=False)
    
    # Print the length of output_data
    print(len(output_data))