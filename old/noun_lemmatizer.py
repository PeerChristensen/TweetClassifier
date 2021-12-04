# import spacy

# nlp = spacy.load('en_core_web_sm', exclude=["ner", "parser"])


# def noun_lemmatizer(text):
#   tokens = nlp(text)
#   tokens = [word.lemma_.lower() for word in tokens if word.pos_ == "NOUN"]
#   return tokens

# tokenizer_name = trial.suggest_categorical(
#   "tokenizer", ["simple_tokenizer", "noun_lemmatizer"]
# )
# tokenizer_map = {
#   "simple_tokenizer": None,
#   "noun_lemmatizer": noun_lemmatizer,
# }

# "vct__tokenizer": tokenizer_map[tokenizer_name],