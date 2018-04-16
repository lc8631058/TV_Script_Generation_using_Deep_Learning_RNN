import os
import pickle


def load_data(path):
    """
    Load Dataset from File
    """
    # os.path.join: Join one or more path components intelligently.
    # 是在拼接路径的时候用的。举个例子，
    # os.path.join(“home”, "me", "mywork")
    # 在Linux系统上会返回
    # “home/me/mywork"
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data


def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)
    
    # Ignore notice, since we don't use it for analysing the data
    text = text[81:]

    # 创建Tonkenize dict
    token_dict = token_lookup()
    for key, token in token_dict.items():
        # replace(old, new): replace old with new. 用format主要目的是替换后前后加空格
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))
