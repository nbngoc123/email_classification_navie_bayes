import string

#################################################################
#                                                               #
#                  TEXT PREPROCESSING PIPELINE                  #
#                                                               #
#################################################################
#                                                               #
#                        [ Raw Message ]                        #
#                             |                                 #
#                             v                                 #
#                       1. Lowercasing                          #
#                             |                                 #
#                             v                                 #
#                 2. Punctuation Removal                        #
#                             |                                 #
#                             v                                 #
#                      3. Tokenization                          #
#                             |                                 #
#                             v                                 #
#                   4. Stopword Removal                         #
#                             |                                 #
#                             v                                 #
#                        5. Stemming                            #
#                             |                                 #
#                             v                                 #
#                      [ Processed Data ]                       #
#                                                               #
#################################################################
