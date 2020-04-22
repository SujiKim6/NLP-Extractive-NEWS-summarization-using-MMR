import mmr_summarizer as mmr
import sys
import json
import os
import re
import sentence
import nltk

def processArticles(article):
    content = " ".join(article)

    # replace all types of quotations by normal quotes
    clean_content = re.sub("\n"," ",content)
    clean_content = re.sub("\"","\"",content)
    clean_content = re.sub("''","\"",content)
    clean_content = re.sub("``","\"",content)
    clean_content = re.sub(" +"," ",content)
    
    # segment data into a list of sentences
    sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
    lines = sentence_token.tokenize(clean_content.strip())

    # setting the stemmer
    sentences = []
    porter = nltk.PorterStemmer()

    # modelling each sentence in file as sentence object
    for line in lines:

		# original words of the sentence before stemming
        originalWords = line[:]
        line = line.strip().lower()

		# word tokenization
        sent = nltk.word_tokenize(line)
		
		# stemming words
        stemmedSent = [porter.stem(word) for word in sent]		
        stemmedSent = filter(lambda x: x!='.'and x!='`'and x!=','and x!='?'and x!="'" 
			and x!='!' and x!='''"''' and x!="''" and x!="'s", stemmedSent)
		
		# list of sentence objects
        if stemmedSent != []:
            sentences.append(sentence.sentence("",list(stemmedSent), originalWords))				

    return sentences

def summaryArticles(sentences, _id, yes, no, multiple, folder):
    IDF_w = mmr.IDFs(sentences)
    TF_IDF_w = mmr.TF_IDF(sentences)
    query = mmr.buildQuery(sentences, TF_IDF_w, 10)

    best1sentence = mmr.bestSentence(sentences, query, IDF_w)

    summary = mmr.makeSummary(sentences, best1sentence, query, 100, 0.5, IDF_w)

    final_summary = ""
    for sent in summary:
        final_summary = final_summary + sent.getOriginalWords() + "\n"
    final_summary = final_summary[:-1]
    
    folder_name = folder
    folder_name += "/yesno_results" if not multiple else "/multi_results"
    file_name = _id
    if yes: file_name += "_yes"
    elif no: file_name += "_no"

    results_folder = os.path.dirname(os.getcwd())+ "/" + folder_name

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    with open(os.path.join(results_folder,(file_name + ".txt")),"w") as fileOut: 
        fileOut.write(final_summary)

if __name__ == "__main__":
    # fileName = sys.argv[1]
    main_folder_path = os.path.dirname(os.getcwd()) + "/articles/"

    for fileName in os.listdir(main_folder_path):
        folder = "test_articles" if fileName.find("test") > 0 else "train_articles"

        fileName = main_folder_path + fileName

        with open(fileName, mode="r") as f:
            QA_dict = json.load(f)

        yesno_questions = QA_dict['yesno_questions']
        multichoice_questions = QA_dict['multichoice_questions']

        for each_yesno in yesno_questions:
            question_id = each_yesno['id']

            # Summary Yes articles
            yes_articles = each_yesno['yes_question_articles']
            yes_sentences = []
            for each_yes in yes_articles:
                yes_sentences = yes_sentences + processArticles(each_yes['content'])

            summaryArticles(yes_sentences, question_id, True, False, False, folder)

            # Summary No articles
            no_articles = each_yesno['no_question_articles']
            no_sentences = []
            for each_no in no_articles:
                no_sentences = no_sentences + processArticles(each_no['content'])
            summaryArticles(no_sentences, question_id, False, True, False, folder)

        for each_multi in multichoice_questions:
            question_id = each_multi['id']

            articles = each_multi['question_articles']
            sentences = []
            for each_art in articles:
                sentences = sentences + processArticles(each_art['content'])

            summaryArticles(sentences, question_id, False, False, True, folder)