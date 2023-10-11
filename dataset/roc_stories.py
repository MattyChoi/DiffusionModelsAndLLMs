"""Modified version of https://huggingface.co/datasets/adamlin/roc_story/blob/main/roc_story.py"""
from typing import Set

import pandas as pd
from sklearn.model_selection import train_test_split

import dataset

_URL = [
    "https://ytlin.s3.ap-northeast-1.amazonaws.com/data/huggingface_datasets/ROCStories/ROCStories2016.csv",
    "https://ytlin.s3.ap-northeast-1.amazonaws.com/data/huggingface_datasets/ROCStories/ROCStories2017.csv",
]


class RocStories(dataset.GeneratorBasedBuilder):
    VERSION = dataset.Version("1.0.0")
    BUILDER_CONFIGS = [dataset.BuilderConfig(name="default")]
    # DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = dataset.Features(
            {
                # origianl features
                "storyid": dataset.Value("string"),
                "storytitle": dataset.Value("string"),
                "sentence1": dataset.Value("string"),
                "sentence2": dataset.Value("string"),
                "sentence3": dataset.Value("string"),
                "sentence4": dataset.Value("string"),
                "sentence5": dataset.Value("string"),
                # model-specific
                "text": dataset.Value("string"),
            }
        )
        return dataset.DatasetInfo(features=features)

    def _split_generators(self, dl_manager: dataset.DownloadManager):
        data_paths = self.dl_manager.download_and_extract(_URL)
        df = pd.concat([pd.read_csv(data_path) for data_path in data_paths])
        storyids = df["storyid"].to_list()

        train_storyids, test_storyids = train_test_split(
            storyids, test_size=0.1, random_state=42
        )
        train_storyids = set(train_storyids)
        test_storyids = set(test_storyids)

        return [
            dataset.SplitGenerator(
                name=dataset.Split.TRAIN,
                gen_kwargs={"df": df, "story_ids": train_storyids},
            ),
            dataset.SplitGenerator(
                name=dataset.Split.VALIDATION,
                gen_kwargs={"df": df, "story_ids": test_storyids},
            ),
            dataset.SplitGenerator(
                name=dataset.Split.TEST,
                gen_kwargs={"df": df, "story_ids": test_storyids},
            ),
        ]

    def _generate_examples(self, df: pd.DataFrame, story_ids: Set):
        id_ = -1
        for row in df.to_dict(orient="records"):
            if row["storyid"] not in story_ids:
                continue
            sentences = [
                row["sentence1"],
                row["sentence2"],
                row["sentence3"],
                row["sentence4"],
                row["sentence5"],
            ]
            row["text"] = f" ".join(sentences)
            id_ += 1
            yield id_, row
