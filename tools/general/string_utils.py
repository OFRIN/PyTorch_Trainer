
def is_english(string):
    e_count = 0
    n_count = 0
    s_count = 0
    unknown = []

    for c in string:
        # if ord('가') <= ord(c) <= ord('힣'):
        #     k_count += 1
        if ord('a') <= ord(c) <= ord('z'):
            e_count += 1
        elif ord('A') <= ord(c) <= ord('Z'):
            e_count += 1

        elif ord('0') <= ord(c) <= ord('9'):
            n_count += 1

        elif ord(' ') == ord(c) or ord(',') == ord(c):
            s_count += 1
        elif ord('(') == ord(c) or ord(')') == ord(c):
            s_count += 1
        elif ord('.') == ord(c) or ord('\'') == ord(c):
            s_count += 1
        elif ord('-') == ord(c):
            s_count += 1

        else:
            unknown.append(c)

    return len(unknown) == 0

