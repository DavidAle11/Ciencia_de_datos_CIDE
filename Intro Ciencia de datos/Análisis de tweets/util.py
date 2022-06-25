### Funciones de apoyo

import os
import json
import math
import re


def remove_hashtags(txt):
    """Replace URLs found in a text string with nothing 
    (i.e. it will remove the URL from the string).

    Parameters
    ----------
    txt : string
        A text string that you want to parse and remove urls.

    Returns
    -------
    The same txt string with url's removed.
    """

    return re.sub(r'[@#]\w+', '', txt)

def eliminar_nooalfanum(texto):
    """
    Dada una cadena de caracteres, retira todos los caracteres 
    #no-alfanuméricos (utilizando la definición Unicode de alfanumérico)
    Input: Un string
    Output: un string que remplaza texto
    """
    import re
    return re.compile(r'\W+', re.UNICODE).split(texto)

def quitar_palabrasrep(listaPalabras, palabras_rep):
    """
    Dada una lista de palabras, retira cualquiera que esté
    en la lista de palabras funcionales.
    Input: Un string
    Output: un string que remplaza texto
    """
    return [w.lower() for w in listaPalabras if w.lower() not in palabras_rep]

