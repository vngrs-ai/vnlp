def load_file(filename):

    file = open(filename, "r", encoding = 'utf-8')
    if file.mode == 'r':
        contents = file.read()

    return contents


def write_file(list_of_tokens, filename):

    file = open(filename, "a+", encoding = 'utf-8')
    size = len(list_of_tokens)
    for i in range(size):
        file.write("%s\r\n" % (list_of_tokens[i]))


def load_words(wordlist_filename):
    wordlist = list()
    # 'with' can automate finish 'open' and 'close' file
    with open(wordlist_filename, encoding = 'utf-8') as f:
        # fetch one line each time, include '\n'
        for line in f:
            # strip '\n', then append it to wordlist
            wordlist.append(line.rstrip('\n'))
    return wordlist
