licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "files",
    srcs = glob(["*.py", "params/*.json", "benchmark_sgf/*.sgf"]),
)

sh_binary(
    name = "loop_main",
    srcs = ["loop_main.sh"],
    data = [
        ":files",
        "@com_google_minigo//cc:main",
    ],
)
