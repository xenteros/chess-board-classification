import cv2
import pickle
import uuid

from utils import *

EMPTY = "EMPTY"
WHITE = "WHITE"
BLACK = "BLACK"

KING = "K"
QUEEN = "Q"
ROCK = "R"
BISHOP = "B"
KNIGHT = "N"
PAWN = "P"
NONE = ""


def prepare_pickle_with_chessboard_setup(setup_name):
    if setup_name == "0":
        d = {
            "a": [(WHITE, ROCK), (WHITE, PAWN), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE),
                  (BLACK, PAWN),
                  (BLACK, ROCK)],
            "b": [(EMPTY, NONE), (WHITE, PAWN), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE),
                  (BLACK, PAWN),
                  (EMPTY, NONE)],
            "c": [(WHITE, BISHOP), (WHITE, PAWN), (WHITE, KNIGHT), (EMPTY, NONE), (EMPTY, NONE), (BLACK, KNIGHT),
                  (BLACK, PAWN), (EMPTY, NONE)],
            "d": [(WHITE, QUEEN), (EMPTY, NONE), (EMPTY, NONE), (WHITE, PAWN), (EMPTY, NONE), (BLACK, PAWN),
                  (EMPTY, NONE),
                  (EMPTY, NONE)],
            "e": [(EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE), (WHITE, PAWN), (EMPTY, NONE), (EMPTY, NONE),
                  (EMPTY, NONE),
                  (EMPTY, NONE)],
            "f": [(WHITE, ROCK), (EMPTY, NONE), (EMPTY, NONE), (BLACK, PAWN), (EMPTY, NONE), (EMPTY, NONE),
                  (BLACK, KING),
                  (BLACK, BISHOP)],
            "g": [(WHITE, KING), (WHITE, PAWN), (EMPTY, NONE), (EMPTY, NONE), (BLACK, QUEEN), (EMPTY, NONE),
                  (EMPTY, NONE),
                  (BLACK, KNIGHT)],
            "h": [(EMPTY, NONE), (WHITE, PAWN), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE),
                  (BLACK, PAWN),
                  (BLACK, ROCK)]
        }
    elif setup_name == "1":
        d = {
            "a": [(WHITE, ROCK), (WHITE, PAWN), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE),
                  (BLACK, PAWN),
                  (BLACK, ROCK)],
            "b": [(WHITE, KNIGHT), (WHITE, PAWN), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE),
                  (BLACK, PAWN),
                  (BLACK, KNIGHT)],
            "c": [(WHITE, BISHOP), (WHITE, PAWN), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE),
                  (BLACK, PAWN), (BLACK, BISHOP)],
            "d": [(WHITE, QUEEN), (WHITE, PAWN), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE),
                  (BLACK, PAWN),
                  (BLACK, QUEEN)],
            "e": [(WHITE, KING), (WHITE, PAWN), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE),
                  (BLACK, PAWN),
                  (BLACK, KING)],
            "f": [(WHITE, BISHOP), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE),
                  (BLACK, PAWN),
                  (BLACK, BISHOP)],
            "g": [(WHITE, KNIGHT), (WHITE, PAWN), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE),
                  (EMPTY, NONE),
                  (BLACK, KNIGHT)],
            "h": [(WHITE, ROCK), (WHITE, PAWN), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE),
                  (BLACK, PAWN),
                  (BLACK, ROCK)]
        }
    elif setup_name == "2":
        d = {
            "a": [(EMPTY, NONE), (EMPTY, NONE), (WHITE, PAWN), (WHITE, ROCK), (BLACK, ROCK), (BLACK, PAWN),
                  (EMPTY, NONE), (EMPTY, NONE)],
            "b": [(EMPTY, NONE), (EMPTY, NONE), (WHITE, PAWN), (WHITE, KNIGHT), (BLACK, KNIGHT), (BLACK, PAWN),
                  (EMPTY, NONE), (EMPTY, NONE)],
            "c": [(EMPTY, NONE), (EMPTY, NONE), (WHITE, PAWN), (WHITE, BISHOP), (BLACK, BISHOP), (BLACK, PAWN),
                  (EMPTY, NONE), (EMPTY, NONE)],
            "d": [(EMPTY, NONE), (EMPTY, NONE), (WHITE, PAWN), (WHITE, QUEEN), (BLACK, QUEEN), (BLACK, PAWN),
                  (EMPTY, NONE), (EMPTY, NONE)],
            "e": [(EMPTY, NONE), (EMPTY, NONE), (WHITE, PAWN), (WHITE, KING), (BLACK, KING), (BLACK, PAWN),
                  (EMPTY, NONE), (EMPTY, NONE)],
            "f": [(EMPTY, NONE), (EMPTY, NONE), (EMPTY, NONE), (WHITE, BISHOP), (BLACK, BISHOP), (EMPTY, NONE),
                  (EMPTY, NONE), (EMPTY, NONE)],
            "g": [(EMPTY, NONE), (EMPTY, NONE), (WHITE, PAWN), (WHITE, KNIGHT), (BLACK, KNIGHT), (BLACK, PAWN),
                  (EMPTY, NONE), (EMPTY, NONE)],
            "h": [(EMPTY, NONE), (EMPTY, NONE), (WHITE, PAWN), (WHITE, ROCK), (BLACK, ROCK), (BLACK, PAWN),
                  (EMPTY, NONE), (EMPTY, NONE)]
        }

    return d


def save_obj(obj, name):
    with open('setups/' + name + "/setup" + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('setups/' + name + "/setup" + '.pkl', 'rb') as f:
        return pickle.load(f)


def distribute_fields_into_directories(fields, chessboard_setup):
    for key in chessboard_setup:
        col = chessboard_setup[key]
        for i in range(8):
            field = key + str(i + 1)
            color, piece = col[i]
            if color == EMPTY:
                ensure_dir(EMPTY)
                name = str(uuid.uuid4()).replace('-', '')
                path = EMPTY + "/" + name + ".jpg"
                cv2.imwrite(path, fields[field])
            else:
                ensure_dir(color + "/" + piece)
                name = str(uuid.uuid4()).replace('-', '')
                path = color + "/" + piece + "/" + name + ".jpg"
                cv2.imwrite(path, fields[field])
