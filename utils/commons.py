def check_entity_continue(tag):
    label = False
    if tag.startswith("I"):
        label = True
    return label


def ner_format_result(char_list, tag_list):
    entities, entity = [], []
    for idx, tag in enumerate(tag_list):
        check_label = check_entity_continue(tag) if idx > 0 else False
        if not check_label and entity:
            entities.append(entity)
            entity = []
        entity.append([idx, char_list[idx], tag])
    entities.append(entity)

    results = []
    for entity in entities:
        if entity[0][2].startswith("B"):
            results.append(
                {
                    "begin": entity[0][0] + 1,
                    "end": entity[-1][0] + 1,
                    "words": "".join([i[1] for i in entity]),
                    "ner_type": entity[0][2].split("-")[1]
                }
            )
    return results