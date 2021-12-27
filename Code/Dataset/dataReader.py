from numpy import genfromtxt

def csv(filename):
    return genfromtxt(filename, delimiter=',',dtype=None,encoding=None)

# from Dataset import dataReader as reader
#
# # x = reader.csv("../../Dataset/annotated Cytometrie/Labeled_Gated F15-631.csv") example use of reader
# # print(x)