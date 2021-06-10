#!/usr/bin/env python3
from genericpath import isdir
from os import listdir
from os.path import isdir, join

HEADERS = []
SOURCES = []
def find_code(dir):
    files = listdir(dir)
    for f in files:
        if f[0] == "." or f == "build":
            continue
        if isdir(join(dir, f)):
            find_code(join(dir, f))
        else:
            if ".h" in f or ".hpp" in f:
                HEADERS.append(join(dir, f))
            elif ".c" in f or ".cc" in f or ".cpp" in f or ".cu" in f:
                SOURCES.append(join(dir, f))

find_code(".")

buff = "HEADERS ="
for h in HEADERS:
    buff += " " + h.replace("./", "")
buff += "\n\n"

obj_list = ""
for s in SOURCES:
    s = s.replace("./", "")
    module_name = s.split(".")[0]
    src_buff = module_name + ".o: " + s + " $(HEADERS)\n\t"
    src_buff += "nvcc -c " + s + " -o " + module_name + ".o\n"
    buff += src_buff
    obj_list += module_name + ".o "


buff += "nvbandwidth: " + obj_list + "\n\t"
buff += "nvcc " + obj_list + "-o nvbandwidth"

print(buff)
