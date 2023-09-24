import difflib
import regtag
import random


def merge_span(words, tags):
    spans, span_tags = [], []
    current_tag = 'O'
    span = []
    for w, t in zip(words, tags):
        w = w.strip(":-")
        if len(w) == 0:
            continue
        t_info = t.split('-')
        if t_info[-1] != current_tag or t_info[0] == 'B':
            if len(span) > 0:
                spans.append(' '.join(span))
                span_tags.append(current_tag)
            span = [w]
            current_tag = t_info[-1]
        else:
            span.append(w)
    if len(span) > 0:
        spans.append(' '.join(span))
        span_tags.append(current_tag)
    return spans, span_tags


def make_spoken(text, do_split=True):
    src, tgt = [], []
    if do_split:
        chunk_size = random.choice(list(range(0, 10)) + list(range(10, 35)) * 4)
        if chunk_size > 0:
            text = random.choice(split_chunk_input(text, chunk_size))
        else:
            text = ''
    words, word_tags = merge_span(*regtag.tagging(text))
    for span, t in zip(words, word_tags):
        if t == 'O':
            for w in span.split():
                w = w.strip('/.,?!').lower()
                if len(w) > 0:
                    src.append(w)
                    tgt.append(w)
                if random.random() < 0.01:
                    random_value = regtag.augment.get_random_span()
                    tgt.append(random_value[0])
                    src.append(random_value[1].lower())
        else:
            random_value = regtag.augment.get_random_span(t, span.lower())
            tgt.append(random_value[0])
            src.append(random_value[1].lower())

    if len(src) == 0:
        tgt, src = regtag.get_random_span()
        src = [src]
        tgt = [tgt]

    return src, tgt


def split_chunk_input(raw_text, chunk_size):
    input_words = raw_text.strip().split()
    clean_data = [input_words[i:i + chunk_size] for i in range(0, len(input_words), chunk_size)]
    if len(clean_data) > 1:
        clean_data = [" ".join(clean_data[i] + clean_data[i + 1]) for i in range(len(clean_data) - 1)]
    else:
        clean_data = [" ".join(clean_data[0])]
    return clean_data


def split_chunk_input(raw_text, chunk_size=40, overlap=10):
    input_words = raw_text.strip().split()
    part_per_chunk = chunk_size // overlap
    clean_data = [input_words[i:i + overlap] for i in range(0, len(input_words), overlap)]
    if len(clean_data) > 1 and part_per_chunk > 1:
        merge_data = []
        for i in range(0, len(clean_data) - 1, part_per_chunk - 1):
            merge_data.append(' '.join([y for x in clean_data[i:i + part_per_chunk] for y in x]))
    else:
        merge_data = [" ".join(clean_data[0])]
    return merge_data


def merge_two_chunk(chunk_1, chunk_2, overlap, debug=False):
    def extract_phrase_word(phrase):
        if phrase.startswith('<mask>'):
            return phrase[7:].split('](')[1][:-1].split()
        else:
            return [phrase]

    def has_tag(phrase):
        if phrase.startswith('<') and phrase.endswith(')'):
            return True
        return False

    def extract_compete_region(list_phrases, is_head):
        if is_head:
            list_phrases = list_phrases[::-1]
        compete = []
        remain = []
        handle_count = 0
        for phrase in list_phrases:
            phrase_word = extract_phrase_word(phrase)
            if len(phrase_word) + handle_count <= overlap:
                compete.append(phrase)
                handle_count += len(phrase_word)
            else:
                if handle_count < overlap:
                    remain_compete_count = overlap - handle_count
                    remain.append(phrase)
                    if not is_head:
                        compete.extend(["<delete>({})".format(item) for item in phrase_word[:remain_compete_count]])
                    else:
                        compete.extend(
                            ["<delete>({})".format(item) for item in phrase_word[::-1][:remain_compete_count]])
                    handle_count = overlap
                else:
                    remain.append(phrase)
        if is_head:
            compete = compete[::-1]
            remain = remain[::-1]
        return remain, compete

    def is_equal(phrase_1, phrase_2):
        if phrase_1 == phrase_2:
            return True
        if extract_phrase_word(phrase_1) == extract_phrase_word(phrase_2):
            if phrase_1.startswith('<mask>') and phrase_2.startswith('<mask>'):
                return True
        return False

    def merge_compete(list_1, list_2):
        idx_list_1, idx_list_2, combine_phrases = [], [], []
        mark_term_complete = []
        list_raw = [extract_phrase_word(item) for item in list_1]
        list_raw = [y for x in list_raw for y in x]
        for idx, phrase in enumerate(list_1):
            idx_list_1.extend([idx] * len(extract_phrase_word(phrase)))
        for idx, phrase in enumerate(list_2):
            idx_list_2.extend([idx] * len(extract_phrase_word(phrase)))
        # print(idx_list_1, idx_list_2)
        for idx, (idx_1, idx_2) in enumerate(zip(idx_list_1, idx_list_2)):
            if list_1[idx_1].startswith('<delete>') or list_2[idx_2].startswith('<delete>'):
                continue
            elif is_equal(list_1[idx_1], list_2[idx_2]):
                # print(list_1[idx_1])
                if '1_{}'.format(idx_1) not in mark_term_complete and '2_{}'.format(idx_2) not in mark_term_complete:
                    if idx <= overlap//2:
                        combine_phrases.append(list_1[idx_1])
                        mark_term_complete.append('1_{}'.format(idx_1))
                    else:
                        combine_phrases.append(list_2[idx_2])
                        mark_term_complete.append('2_{}'.format(idx_2))
            else:
                combine_phrases.append(list_raw[idx])
                mark_term_complete.extend(['1_{}'.format(idx_1), '2_{}'.format(idx_2)])
        # print(mark_term_complete)
        return combine_phrases

    remain_1, compete_1 = extract_compete_region(chunk_1, is_head=True)
    remain_2, compete_2 = extract_compete_region(chunk_2[1:-1], is_head=False)
    compromise = merge_compete(compete_1, compete_2)

    if debug:
        print(remain_1, '\n', compete_1)
        print('-----------------------')
        print(compete_2, '\n', remain_2)
        print('-----------------------')
        print(compromise, '\n\n')

    return remain_1 + compromise + remain_2


def merge_chunk_pre_norm(list_chunks, overlap, debug=False):
    if len(list_chunks) == 0:
        return []
    if len(list_chunks) == 1:
        return list_chunks[0][1:-1]
    current_chunk = list_chunks[0][1:-1]
    for tmp_chunk in list_chunks[1:]:
        current_chunk = merge_two_chunk(current_chunk, tmp_chunk, overlap, debug=debug)
    return current_chunk


def equalize(s1, s2):
    l1 = s1.split()
    l2 = s2.split()
    res1 = []
    res2 = []
    combine = []
    prev = difflib.Match(0, 0, 0)
    for match in difflib.SequenceMatcher(a=l1, b=l2).get_matching_blocks():
        if prev.a + prev.size != match.a:
            for i in range(prev.a + prev.size, match.a):
                res2 += ['_' * len(l1[i])]
            res1 += l1[prev.a + prev.size:match.a]

            for i in l1[prev.a + prev.size:match.a]:
                if len(combine) < len(l1) // 2:
                    print(l1[prev.a + prev.size:match.a])
                    combine.append(i)
        if prev.b + prev.size != match.b:
            for i in range(prev.b + prev.size, match.b):
                res1 += ['_' * len(l2[i])]
            res2 += l2[prev.b + prev.size:match.b]

            for i in l2[prev.b + prev.size:match.b]:
                if len(combine) >= len(l2) // 2:
                    print(l2[prev.b + prev.size:match.b])
                    combine.append(i)
        res1 += l1[match.a:match.a + match.size]
        res2 += l2[match.b:match.b + match.size]
        combine += l2[match.b:match.b + match.size]
        prev = match
    return ' '.join(res1), ' '.join(res2), combine


def count_overlap(words_1, words_2):
    # print(words_1, words_2)
    assert len(words_1) == len(words_2)
    len_overlap = 0
    for match in difflib.SequenceMatcher(a=words_1, b=words_2).get_matching_blocks():
        len_overlap += match.size

    # for w1, w2 in zip(words_1, words_2):
    #     if w1 == w2:
    #         len_overlap += 1
    return len_overlap


def find_overlap_chunk(txt_1, txt_2):
    # print(txt_1)
    # print(txt_2)
    window_view = 1
    idx_1 = len(txt_1) - window_view
    idx_2 = window_view
    over_lap = 0
    current_best_idx_1 = len(txt_1)
    current_best_idx_2 = 0

    while window_view <= len(txt_1) and window_view <= len(txt_2):
        current_overlap = count_overlap(txt_1[idx_1:], txt_2[:idx_2])
        print(current_overlap)
        if over_lap < current_overlap:
            over_lap = current_overlap
            current_best_idx_1 = idx_1
            current_best_idx_2 = idx_2
        window_view += 1
        idx_1 = len(txt_1) - window_view
        idx_2 = window_view
        # else:
        #     break
    print('----->', txt_1[current_best_idx_1:], txt_2[:current_best_idx_2])
    return txt_1[current_best_idx_1:], txt_2[:current_best_idx_2]


def concat_chunks(list_chunks):
    concat_string = list_chunks[0].split()
    for i in range(1, len(list_chunks)):
        remain_string = list_chunks[i].split()
        s1, s2 = find_overlap_chunk(concat_string, remain_string)
        s1 = ' '.join(s1)
        s2 = ' '.join(s2)
        _, _, overlap_merged = equalize(s1, s2)
        merge_len = len(s1.split())

        concat_string = concat_string[:len(concat_string) - merge_len] + overlap_merged + remain_string[merge_len:]

    concat_string = ' '.join(concat_string)
    return concat_string
