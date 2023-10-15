def make_features(df, task):
    return df["video_name"], get_output(df, task)


def get_output(df, task):
    if task == "is_comic_video":
        return df["is_comic"]
    elif task == "is_name":
        return df["is_name"]
    elif task == "find_comic_name":
        return df["comic_name"]

    raise ValueError("Unknown task")
