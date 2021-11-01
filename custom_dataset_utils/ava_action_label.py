
def get_action_dict(action_list_path):
    with open(action_list_path) as f:
        data = f.read().replace("\n", "")
        data_split = data.split("}")

        action_dict = {"PERSON_MOVEMENT": [],
                       "OBJECT_MANIPULATION": [],
                       "PERSON_INTERACTION": []}
        for ds in data_split:
            ds = ds.replace("name", "'name'").replace("label_id", ",'label_id'").replace("label_type", ",'label_type'")
            ds = ds.replace("PERSON_MOVEMENT", "'PERSON_MOVEMENT'")
            ds = ds.replace("OBJECT_MANIPULATION", "'OBJECT_MANIPULATION'")
            ds = ds.replace("PERSON_INTERACTION", "'PERSON_INTERACTION'") + "}"
            ds = ds.replace("label {", "{")
            if len(ds) < 5:
                continue
            tmp_dict = eval(ds)
            action_dict[tmp_dict["label_type"]].append(tmp_dict["name"])
    return action_dict


if __name__ == "__main__":
    pass