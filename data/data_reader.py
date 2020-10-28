import os
import numpy as np
import pandas as pd


class MovieLensReader(object):
    def __init__(self):
        self.base_path = os.path.dirname(__file__)
        self.movies_file = os.path.join(self.base_path, "./ml-1m/movies.dat")
        self.users_file = os.path.join(self.base_path, "./ml-1m/users.dat")
        self.rating_file = os.path.join(self.base_path, "./ml-1m/ratings.dat")
        self.idx_movie_dict, self.movie_idx_dict = self.get_mappings(self.movies_file)
        self.idx_user_dict, self.user_idx_dict = self.get_mappings(self.users_file)


    def get_mappings(self, input_file):
        entity_list = []
        with open(input_file, 'r') as f:
            for line in f:
                entity = line.strip().split("::")[0]
                entity_list.append(entity)
        idx_entity_dict = dict([int(idx), int(entity)] for idx, entity in enumerate(entity_list))
        entity_idx_dict = dict([int(entity), int(idx)] for idx, entity in idx_entity_dict.items())
        return idx_entity_dict, entity_idx_dict


    def load_rating_as_binary(self, threshold=4):
        x_list = []
        y_list = []
        with open(self.rating_file, 'r') as f:
            for line in f:
                ss = line.strip().split("::")
                if len(ss) != 4:
                    continue
                userid, itemid, score, timestamp = ss
                userid = self.user_idx_dict[int(userid)]
                itemid = self.movie_idx_dict[int(itemid)]
                if int(score) > threshold:
                    x_list.append([userid, itemid])
                    y_list.append(1)
                else:
                    x_list.append([userid, itemid])
                    y_list.append(0)
        return x_list, y_list


movie_lens_reader = MovieLensReader()





