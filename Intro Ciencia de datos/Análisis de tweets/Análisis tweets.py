# Funciones análisis twitters

## Descargar bibliotecas 
from util import * 
import os
import json
import math
from math import log
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

## Generar variable global stopwords

stopwords = stopwords.words("spanish")
stopwords += ["rt", "https", "t", "co"]

def count_tokens(token_list):
    """
    ** Determina las frecuencias de los tokens
    
    Parámetros: token_list (lista): lista que contiene los tokens en cadena
    Returns: diccionario (diccionario): regresa en clave los tokens y en valor la frecuencia de estos
    """
    diccionario = {} # se inicializa el diccionario de frecuencias
    
    for token in token_list: # se itera por la lista
        if token in diccionario: # se verifica si el token ya esta en el diccinario
            diccionario[token] += 1 # si ya esta, la frecuencia aumenta en 1
        else: # si no esta
            diccionario[token] = 1 # aparece por primera vez, entonces la frecuencia inicia en 1
   
    return (diccionario)

def find_top_k(token_list, k):
    """
    ** Determina el top-k de tokens
    
    Parámetros: token_list (lista): lista que contiene los tokens en cadena
                k (entero): hace referencia al top k
    Returns: top_k (lista): lista de los k-tokens que ocurren con mayor frecuencia en token_list,
                            orden descendente
    """
    lista = sorted(list(count_tokens(token_list).items()), key = lambda x: x[1], reverse = True) # lista donde se guardarán (token, frecuencia) en orden descendente
    top_k = [lista[i][0] for i in range(k)] # agregamos a top_k las k primeras tuplas que son los k tokens que tienen mayor frecuencia
    
    return (top_k)

def find_min_count(token_list, k):
    """
    ** Determina que tokens ocurren al menos k veces
    
    Parámetros: token_list (lista): lista que contiene los tokens en cadena
                k (entero): hace referencia al top k
    Returns: conjunto (conjunto): contiene a los tokens que aparecen al menos k veces
    """

    lista_frecuencias = list(count_tokens(token_list).items()) # diccionario de token y frecuencia convertido en lista
    conjunto = set([i[0] for i in lista_frecuencias if i[1] >= k]) # se agregan a conjunto los tokens que aparecen al menos k veces

    return (conjunto)

def frecuencia_basica_documento(docs):
    """
    ** Determina la frecuencia básica de documento inverso de un término en una colección de documentos
    
    Parámetros: docs (lista): colección de documentos que también son listas
                threshold (float): umbral
    Returns: idf_dic (diccionario): diccionario de los tokens con su respectiva frecuencia inversa
    """
    # cálculo de la inversa
    lista_auxiliar = [y for x in docs for y in x] # junta en una sola lista todos los tokens que vienen en los documentos
    tokens = set(lista_auxiliar) # extraemos los diferentes tipos de tokens que puede haber en lista auxiliar
    apariciones = [] # es la lista en donde se guarda la cantidad de documentos en donde aparece cierto token
    for token in tokens: # iteramos sobre los diferentes tipos de token que pueden existir
        contador = 0 # contador nos dirá en cuantos documentos aparece cierto token, se inicializa en 0
        for documento in docs: # se itera sobre cada documento
            if token in documento: # si el token llegara a estar en cierto documento
                contador += 1 # el contador aumenta en 1
        apariciones.append(contador) # se agrega la cantidad de documentos en los que aparecio el token
 
    apariciones = list(zip(tokens, apariciones)) # apariciones pasa a ser un zipeado entre el token y la catidad de documentos en los que aparecio el token
    N = len(docs) # es la cantidad de documentos en total
    idf_dic = {} # se crea un diccionario que nos guarde los diferentes tipos de tokens que puede haber con su frecuencia inversa de documentos respectiva
    
    for token, aparicion in apariciones: # se itera sobre apariciones (tokens y cantidad de docuementos en donde aparece el token)  
        idf_dic[token] = log(N/aparicion) # se agrega al diccionario el token con su respectiva frecuencia inversa
    
    return(idf_dic)


def find_salient(docs, threshold):
    """
    ** Determina los tokens destacados
    
    Parámetros: docs (lista): colección de documentos que también son listas
                threshold (float): umbral
    Returns: tokens_destacados (lista): lista de los tokens destacados
    """
    
    tokens_destacados = [] 
    idf = frecuencia_basica_documento(docs)
    
    for documento in docs: # se itera en cada documento 
        if documento: # si el documento no es una lista vacia
            diccionario = count_tokens(documento) # diccionario en donde están los tokens y sus frecuencias en cierto documento
            ft_prima = max(diccionario.values()) # es la frecuencia del token con mayor frecuencia
            si_pasa = set() # conjunto en donde se guardan los tokens que si son destacados de cierto documento
            for token in diccionario: # se itera sobre todas las claves del diccionario de frecuencias
                tf = 0.5 + 0.5*(diccionario[token]/ft_prima) # se calcula el término frecuencia
                tfidf = tf * idf[token] # se calcula el producto del término frecuencia y la frecuencia inversa como indicador de si pasa o no
                if tfidf > threshold: # si el indicador es mayor que el umbral
                    si_pasa.add(token) # se agrega el token al conjunto de los tokens que si destacan
                    
            tokens_destacados.append(si_pasa) # se agrega el conjunto de tokens que si destacaron a la lista de tokens_destacados y esto se hace para cada documento
            
        else: # en caso en que el documento sea una lista vacia
            tokens_destacados.append(set()) # se agrega un conjunto vacio a tokens_destacados

    return (tokens_destacados) 

def find_top_k_entities(tweets, entity_desc, k):
    """
    ** Determina las top-k entidades
    
    Parámetros: tweets (lista): lista de información a analizar
                entity_desc (tupla): contiene la informacion de clave y subclave y booleano
                k (entero): hace referencia al top k
    Returns: top (lista): se regresan los top k tokens 
    """
   
    clave = entity_desc[0] # clave para diccionario
    subclave = entity_desc[1] # subclave para diccionario
    lista = [tweet['entities'][clave][0][subclave].lower() for tweet in tweets if tweet['entities'][clave]] # lista de textos de hashtags
    #lista1 = [dic[subclave] for tweet in tweets for dic in tweet['entities'][clave]]
    '''
    NOTA: En lista1 se iteran sobre todos los diccionarios que puede haber en la lista tweet['entities'][clave]
          y no solo en en primer diccionario si es que este existiera como se hace en lista.
          lo hicimos solo iterando sobre el primer diccionario porque nos dimos cuenta que en los
          resultados de la tarea que ustedes proponen, coinciden si se hace como en lista.
          Para obtener la informacion completa favor de descomentar lista 1 y colocarla en find_top_k
    '''
    top = find_top_k(lista, k) # se encuentran los primeros k
    
    return (top)

def find_min_count_entities(tweets, entity_desc, min_count):
    """
    **Determina las entidades que aparecen un número mínimo de veces
    Parámetros: tweets (lista): lista de información a analizar
                entity_desc (tupla): contiene la informacion de clave y subclave y booleano
                min_count (entero): número minimo de veces que tiene que aparecer un tweet
    Returns: conjunto (conjunto): contiene a los tokens que aparecen al menos min_count veces
    """
    clave = entity_desc[0] # clave para diccionario
    subclave = entity_desc[1] # subclave para diccionario
    lista = [tweet['entities'][clave][0][subclave].lower() for tweet in tweets if tweet['entities'][clave]] # lista de los names de user_mention
    #lista1 = [dic[subclave] for tweet in tweets for dic in tweet['entities'][clave]]
    '''
    NOTA: En lista1 se iteran sobre todos los diccionarios que puede haber en la lista tweet['entities'][clave]
          y no solo en en primer diccionario si es que este existiera como se hace en lista.
          lo hicimos solo iterando sobre el primer diccionario porque nos dimos cuenta que en los
          resultados de la tarea que ustedes proponen, coinciden si se hace como en lista.
          Para obtener la informacion completa favor de descomentar lista 1 y colocarla en find_min_count
    '''
    conjunto = find_min_count(lista, min_count) # conjunto que contiene los tokens que aparecen al menos min_count veces
    
    return (conjunto)


def find_top_k_ngrams(tweets, ngram, k):
    """
    ** Determina el top-k de n gramas
    
    Parámetros: tweets (lista): lista de información a analizar
                ngram (entero): entero que hace referencia a n-gramos
                k (entero): hace referencia al top k
    Returns: top_k_grams (lista): lista del top k de los n gramos 
	"""
    n_gramos = [] # se acumularán los gramos de cada tweet en tweets
    
    for i in range(len(tweets)): # comienza iteracion
        tweet_clean = remove_hashtags(tweets[i]["text"]) 
        tweet_text = eliminar_nooalfanum(tweet_clean)
        tweet_final = quitar_palabrasrep(tweet_text , stopwords)
        tope = len(tweet_final) - ngram # nos ayuda a encontrar a partir de que elemento se construirá la ultima grama para no salirnos de rango
        n_gramos += [tuple([tweet_final[j + i] for i in range(ngram)]) for j in range(tope + 1)] # se van acumulando las gramas de cada tweet
    
    top_k_grams = find_top_k(n_gramos, k)
    
    return (top_k_grams)


def find_salient_ngrams(tweets, ngram, threshold):
    """
    ** Determina las n-gramas destacados en cada tweet de una colección
    
    Parámetros: tweets (lista): lista de información a analizar
                ngram (entero): entero que hace referencia a n-gramos
                threshold (float): umbral
    Returns: destacados (lista): lista de los n- gramas destacados 
	"""
    lista = [] # se irán acumulando los n gramos de cada tweet aquí
    for tweet in tweets:
        tweet_clean = remove_hashtags(tweet["text"]) 
        tweet_text = eliminar_nooalfanum(tweet_clean)
        tweet_final = quitar_palabrasrep(tweet_text , stopwords)
        tope = len(tweet_final) - ngram # nos ayuda a encontrar a partir de que elemento se construirá la ultima grama para no salirnos de rango
        n_gramos = [tuple([tweet_final[j + i] for i in range(ngram)]) for j in range(tope + 1)] # se construyen los n-gramos de cada tweet
        lista.append(n_gramos) # los vamos acumulando
    
    destacados = find_salient(lista, threshold) # se encuentran los destacados
    
    return(destacados)


def find_salient_ngrams_ingles(tweets, ngram, threshold):
    """
    ** Determina las n-gramas destacados en cada tweet de una colección pero en ingles
    Parámetros: tweets (lista): lista de información a analizar
                ngram (entero): entero que hace referencia a n-gramos
                threshold (float): umbral
    Returns: destacados (lista): lista de los n- gramas destacados 
	"""
    lista = [] # se irán acumulando los n gramos de cada tweet aquí
    for tweet in tweets:
        tweet_final = tweet['text'].split()
        tope = len(tweet_final) - ngram # nos ayuda a encontrar a partir de que elemento se construirá la ultima grama para no salirnos de rango
        n_gramos = [tuple([tweet_final[j + i] for i in range(ngram)]) for j in range(tope + 1)] # se construyen los n-gramos de cada tweet
        lista.append(n_gramos) # los vamos acumulando
    
    destacados = find_salient(lista, threshold) # se encuentran los destacados
    
    return(destacados)

#__________________________________________________________

if __name__ == "__main__":
    
    with open("twitter_NAICM_full.json", "r+") as jsonFile:
        NAICM = json.load(jsonFile)
        
    menu = int(input('Presiona 1 si quiere ver las pruebas, presione 2 si quiere ver la parte final: '))
    
    if menu == 1:
        """
        Si corres este cÃ³digo, las funciones deberÃ­an dar el mismo
        resultado que en la documentaciÃ³n de la tarea
        """

        res1 = count_tokens(["A", "A", "B", "B", "C"])
        print(res1)

        l = ['D', 'B', 'C', 'D', 'D', 'B', 'D', 'C', 'D', 'A']
        res2 = find_top_k(l, 2)
        print(res2)

        res3 = find_min_count(l, 2)
        print(res3)

        coleccion = [['D', 'B', 'D', 'C', 'D', 'C', 'C'], 
 	                 ['D', 'A', 'A'], ['D', 'B'], []]

        res4 = find_salient(coleccion, .4)
        print(res4)

        res5 = find_top_k_entities(NAICM[:50], ("hashtags", "text"), 3)
        print(res5)

        res6 = find_min_count_entities(NAICM[:500], ("user_mentions", "name"), 10)
        print(res6)

        res7 = find_top_k_ngrams(NAICM[:10000], 3, 10)
        print(res7)

        tweets = [ {"text": "the cat in the hat" },
   	               {"text": "don't let the cat on the hat" },
   	               {"text": "the cat's hat" },
   	               {"text": "the hat cat" }]
        
        res8 = find_salient_ngrams_ingles(tweets, 2, 1.3) # usamos esta funcion porque los textos estan en ingles
        print(res8)
    
    elif menu == 2:
        
        tweets = NAICM
        fechas = []

        for i in range(len(tweets)):
            a = tweets[i]['created_at'].split()
            dia = a[0] + ' ' + a[1] + ' ' + a[2]
            if dia not in fechas:
                fechas.append(dia)
        
        confirmacion = 1
        
        while confirmacion == 1:
            
            ejercicio = int(input('Presiona del 1 al 7 según el ejercicio que desee ver: '))
            
            if ejercicio == 1:
                
                print ('\n')
                print ('______________ Ejercicio 1')
                print ('Los 10 principales hashtags por día \n')
        

                diccionario = {}
                for fecha in fechas:
                    diccionario[fecha] = []
                    
                for tweet in (tweets):
                    if tweet['entities']['hashtags']:
                        a = tweet['created_at'].split()
                        dia = a[0] + ' ' + a[1] + ' ' + a[2]
                        for dic in tweet['entities']['hashtags']:
                            diccionario[dia].append(dic['text'])
                k = 10
                for dia in diccionario:
                    res = find_top_k(diccionario[dia], k)  
                    print (dia)
                    print (res)
                    print ('\n')
            
            elif ejercicio == 2: 
                
                print ('\n')
                print ('_____________________ Ejercicio 2')
                print ('Los principales usuarios mencionados por día (que tuvieron al menos 50, 100, 250 y 500 menciones)\n')
                
                diccionario = {}
                for fecha in fechas:
                    diccionario[fecha] = []
            
                for tweet in tweets:
                    if tweet['entities']['user_mentions']:
                        a = tweet['created_at'].split()
                        dia = a[0] + ' ' + a[1] + ' ' + a[2]
                        for dic in tweet['entities']['user_mentions']:
                            diccionario[dia].append(dic['name'])
        
                k = [50, 100, 250, 500]
        
                for i in k:
                    print ('Aparecieron al menos', i, 'veces')
                    for dia in diccionario: 
                        res = find_min_count(diccionario[dia], i)  
                        print ('\t', dia)
                        print ('\t', res)
                        print ('\n')
            
            elif ejercicio == 3: 
                
                print ('\n')
                print ('_____________________ Ejercicio 3')
                print ('Los principales usuarios mencionados por día (que tuvieron al menos 50 menciones) de los tweets que fueron favorited \n')
        
                diccionario = {}
                for fecha in fechas:
                    diccionario[fecha] = []
        
        
                for tweet in tweets:
                    if tweet['favorite_count'] > 0:
                        a = tweet['created_at'].split()
                        dia = a[0] + ' ' + a[1] + ' ' + a[2]
                        for dic in tweet['entities']['user_mentions']:
                            diccionario[dia].append(dic['name'])
                
                k = 50
                # Ninguno aparece 50 veces
        
                print ('Aparecieron al menos', k, 'veces')
                for dia in diccionario: 
                    res = find_min_count(diccionario[dia], 48)  
                    print ('\t', dia)
                    print ('\t', res)
                    print ('\n')
                
            elif ejercicio == 4:

                print ('\n')
                print ('_____________________ Ejercicio 4')
                print ('Los 10 principales 2-grams y 3-grams por día')
                print('\n')
        
                diccionario = {}
                for fecha in fechas:
                    diccionario[fecha] = []
        
        
                for tweet in tweets:
                    a = tweet['created_at'].split()
                    dia = a[0] + ' ' + a[1] + ' ' + a[2]
                    diccionario[dia].append(tweet)
         
                for dia in diccionario: 
                    print ('\t', dia)
                    print ('\t 10 principales 2-grams')
                    res = find_top_k_ngrams(diccionario[dia], 2, 10)  
                    print ('\t', res)
                    print ('\t 10 principales 3-grams')
                    res = find_top_k_ngrams(diccionario[dia], 3, 10)  
                    print ('\t', res)
                    print ('\n')
                
            elif ejercicio == 5:

                print ('\n')
                print ('_____________________ Ejercicio 5')
                print ('Los 10 principales 2-grams y 3-grams de los tweets que fueron favorited')
                print('\n')
        
                diccionario = {}
                for fecha in fechas:
                    diccionario[fecha] = []
        
        
                for tweet in tweets:
                    if tweet['favorite_count'] > 0:
                        a = tweet['created_at'].split()
                        dia = a[0] + ' ' + a[1] + ' ' + a[2]
                        diccionario[dia].append(tweet)
                
                for dia in diccionario: 
                    print ('\t', dia)
                    print ('\t ** 10 principales 2-grams favorited')
                    res = find_top_k_ngrams(diccionario[dia], 2, 10)  
                    print ('\t', res)
                    print ('\t ** 10 principales 3-grams favorited')
                    res = find_top_k_ngrams(diccionario[dia], 3, 10)  
                    print ('\t', res)
                    print ('\n')
                    
            elif ejercicio == 6:

                print ('\n')
                print ('_____________________ Ejercicio 6')
                print ('Los 10 principales 2-grams y 3-grams que fueron retweeteados más de 25 veces (retweet_count mayor a 25)')
                print('\n')
        
                diccionario = {}
                for fecha in fechas:
                    diccionario[fecha] = []
        
                for tweet in tweets:
                    if tweet['retweet_count'] > 25:
                        a = tweet['created_at'].split()
                        dia = a[0] + ' ' + a[1] + ' ' + a[2]
                        diccionario[dia].append(tweet)
                
                for dia in diccionario: 
                    print ('\t', dia)
                    print ('\t ** 10 principales 2-grams favorited')
                    res = find_top_k_ngrams(diccionario[dia], 2, 10)  
                    print ('\t', res)
                    print ('\t ** 10 principales 3-grams favorited')
                    res = find_top_k_ngrams(diccionario[dia], 3, 10)  
                    print ('\t', res)
                    print ('\n')
            
            elif ejercicio == 7:
            
                print ('\n')
                print ('_____________________ Ejercicio 7')
                print ('Los 2-grams y 3-grams salientes con un threshold arriba de .5, 1, 1.5 y 2.5')
                print('\n')
        
                print ('\t ** 2-grams sobresalientes arriba de 0.5, 1.5, 2.5 \n')
                res = find_salient_ngrams(tweets[:50], 2, 0.5)  
                print ('\t', res, '\n')
                res = find_salient_ngrams(tweets[:50], 2, 1.5)  
                print ('\t', res, '\n')
                res = find_salient_ngrams(tweets[:50], 2, 2.5)  
                print ('\t', res, '\n\n')
                print ('\t ** 3-grams sobresalientes arriba de 0.5, 1.5, 2.5 \n')
                res = find_salient_ngrams(tweets[:50], 3, 0.5)  
                print ('\t', res, '\n')
                res = find_salient_ngrams(tweets[:50], 3, 1.5)  
                print ('\t', res, '\n')
                res = find_salient_ngrams(tweets[:50], 3, 2.5)  
                print ('\t', res, '\n')

            else:
                
                print ('ESA OPCIÓN NO EXISTE')
                
            confirmacion = int(input('Desea volver a consultar un ejercicio? presione 1 para si o 2 para no: '))
    
'''
NOTA: En los ejercicios 1, 2 y 3 se extrayerón los datos
      de todos los diccionarios que aparecen en las listas 
      tweet['entities']['hashtags'] y tweet['entities']['user_mentions']
      bajo las subclaves 'text' y 'name'
'''       
