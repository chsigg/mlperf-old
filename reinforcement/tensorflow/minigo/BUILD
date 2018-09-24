licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "files",
    srcs = glob([
        "params/*.json",
        "benchmark_sgf/*.sgf",
    ]),
)

py_binary(
    name = "loop_init",
    srcs = glob(["*.py"]),
    data = [":files"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ]
)

py_binary(
    name = "loop_selfplay",
    srcs = glob(["*.py"]),
    data = [
        ":files",
        "@com_google_minigo//cc:main",
    ],
)

py_binary(
    name = "loop_train_eval",
    srcs = glob(["*.py"]),
    data = [
        ":files",
        "@com_google_minigo//cc:main",
    ],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ]
)

sh_binary(
    name = "loop_main",
    srcs = ["loop_main.sh"],
    data = [
        ":loop_init",
        ":loop_selfplay",
        ":loop_train_eval",
    ],
)
