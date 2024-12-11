from xtuner.utils import DEFAULT_IMAGE_TOKEN


def llava_image_only_map_fn(example):
    # input contains the DEFAULT_IMAGE_TOKEN only
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            assert DEFAULT_IMAGE_TOKEN in msg['value']
            input += DEFAULT_IMAGE_TOKEN
        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    return {'conversation': conversation}


# def llava_map_fn(example):
#     messages = example['conversations']
#     input = ''
#     conversation = []
#     while messages and messages[0]['from'] == 'gpt':
#         # Skip the first one if it is from gpt
#         messages = messages[1:]
#     for msg in messages:
#         if msg['from'] == 'human':
#             if DEFAULT_IMAGE_TOKEN in msg['value']:
#                 msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
#                                                     '').strip()
#                 msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
#                 msg['value'] = msg['value'].strip()
#             input += msg['value']

#         elif msg['from'] == 'gpt':
#             conversation.append({'input': input, 'output': msg['value']})
#             input = ''
#         else:
#             raise NotImplementedError
#     return {'conversation': conversation}

# multi - round conversation
def llava_map_fn(example):
    messages = example['conversations']
    conversation = []
    context = ''  

    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]

    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            
            # add human context
            context += f"<|start_header_id|>user<|end_header_id|>\n\n{msg['value']}<|eot_id|>"

        elif msg['from'] == 'gpt':
            # build the conversation
            response = f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg['value']}<|eot_id|>"
            conversation.append({'input': context, 'output': response})
            
            # update the conversation and add gpt's response
            context += response

        else:
            raise NotImplementedError("Message type not supported.")
    
    return {'conversation': conversation}

