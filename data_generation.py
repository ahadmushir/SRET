## Reading files

def get_tokens_start_end(Lines_code, sample_start, sample_end):
    count_start = 0
    count_return = 0

    temp = list()
    for line in Lines_code:
        count_start += 1
        count_return += 1

        if "def " in line:
            temp.append(("def", count_start))
        if "return" in line:
            temp.append(("return", count_return))

    return temp[sample_start:sample_end]
