# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
######################################################################
from pydantic import BaseModel, Field
import numpy as np
import random
import re
import util
import string 
from porter_stemmer import PorterStemmer

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        self.name = 'Bot-tholomew'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        self.user_ratings = np.zeros(len(self.titles))
        
        # chatbot is in the rec-giving stage and user is entering YES or NO
        self.in_rec_stage = False
        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings, threshold=2.5)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "Nice to meet you, I'm Bot-tholomew! Tell me about a movie you watched so I can make some recommendations for you!"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Okay, see you later!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    
    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """Your name is moviebot. You are a movie recommender chatbot. """ +\
        """You can help users find movies they like and provide information about movies."""

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################
    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        if self.llm_enabled:
            response = "I processed {} in LLM Programming mode!!".format(line)
        else:
            response = "I processed {} in Starter (GUS) mode!!".format(line)

        # response options
        one_at_a_time = ["Tell me about one movie at a time, please. My brain is feeling sluggish today.", "Sorry, I can only think about one movie at a time. Please tell me about one movie you watched.", "One movie at a time please! I'm feeling overwhelmed today."]
        nonexistent = ["It doesn't seem like TITLE exists in my database....Tell me about another movie.", "Sorry, I've never heard of TITLE! Tell me about a different movie.", "Hmm, I'm not familiar with TITLE. Tell me about a different movie."]
        positive_echos = ["Good to hear you liked TITLE!", "You liked TITLE? Me too!", "I'm glad you liked TITLE."]
        negative_echos = ["That's too bad that you didn't like TITLE.", "I'm so sorry you didn't like TITLE.", "Yikes, sorry you didn't like TITLE. I also didn't really enjoy it."]
        neutral_echos = ["Hmm, I don't really get what you mean about TITLE....Please tell me if you liked TITLE or not, or tell me about a different movie.", "Sorry, I can't tell if you liked TITLE or not. Can you please clarify?", "I'm confused whether or not you liked TITLE. Please explain further about TITLE."]
        more_movie = ["Tell me about another movie.", "What's another movie you watched, and did you like it?", "What other movie have you seen and what was your opinion?"]
        enough = ["That's enough for me to give a recommendation! Do you want to hear a recommendation? Type YES or NO.", "I know enough about your taste to give you some recommendations now! Do you want to hear a recommendation? Type YES or NO", "Gotcha, thanks for telling me about your taste. Do you want to hear a recommendation? Type YES or NO"]
        rec_sentences = ["You might like: TITLE!", "Based on what you've told me, you should try watching: TITLE!", "Give TITLE a try!"]
        repeat = ["Tell me about more movies for more recommendations, or type :quit to exit.", "Want more recs? Tell me about more movies, or type :quit to exit.", "Still need more recs? Tell me about more movies you've watched, or type :quit to exit."]
        talk_about_movies_only = ["Hmm, I didn't hear you say a movie title. Put the title between double quotation marks. Tell me about a movie you've watched.", "Sorry, I can only talk about movies right now. Make sure to put the title between double quotation marks. Tell me about a movie you've seen recently.", "I didn't catch you mentioning a movie. Put the title between double quotation marks, please. What's your opinion on a movie you've seen recently?"]
        more_rec_question = ["Do you want to hear more recs? Type YES for more recs or NO if you don't want more recs.", "How about another one? Type YES for more recs or NO if you don't want more recs.", "Are you interested in another rec? Type YES for more recs or NO if you don't want more recs."]
        
        ### Process Logic ###
        line = line.lower()
        # in rec-giving stage
        if self.in_rec_stage:
            if line == "yes":
                response += '\n' + random.choice(rec_sentences).replace("TITLE", self.recs[0])
                self.recs = self.recs[1:]
                if self.recs:
                    response += '\n' + random.choice(more_rec_question)
                else:
                    self.in_rec_stage = False
                    response += '\n' + random.choice(repeat)
            elif line == "no":
                self.in_rec_stage = False
                self.recs = []
                response = random.choice(repeat)
            else:
                response = "Please type YES for more recs or NO if you don't want any more recs."

        # in question-asking stage
        else:
            titles = self.extract_titles(line)
            sentiment = self.extract_sentiment(line)
            if len(titles) > 1:
                response = random.choice(one_at_a_time)
            elif len(titles) == 1:
                indices = self.find_movies_by_title(titles[0])
                if not indices:
                    response = random.choice(nonexistent).replace("TITLE", titles[0].title())
                elif sentiment == 0:
                    response = random.choice(neutral_echos).replace("TITLE", titles[0].title())
                else:
                    if sentiment == 1:
                        self.user_ratings[indices[0]] = 1
                        response = random.choice(positive_echos).replace("TITLE", titles[0].title())
                    else:
                        self.user_ratings[indices[0]] = -1
                        response = random.choice(negative_echos).replace("TITLE", titles[0].title())

                    if np.count_nonzero(self.user_ratings) < 5:
                        response += '\n' + random.choice(more_movie)
                    else: # 5 movies given
                        self.in_rec_stage = True
                        response += '\n' + random.choice(enough)
                        rec_nums = self.recommend(self.user_ratings, self.ratings)
                        self.recs = []
                        for index in rec_nums:
                            self.recs.append(self.titles[index][0])
            else:
                response = random.choice(talk_about_movies_only)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text
    
    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """
        return []

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        titles = re.findall(r'"([^"]*)"', preprocessed_input)
        return titles

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """

        ### Title Processing ###
        if title == "":
            return []
        title = title.lower()
        title_list = title.split(' ')
        movie_title = title
        year = -1
        # find year and reconstruct movie title without year
        if title_list[-1][0] == '(':
            year = title_list[-1]
            movie_title = ""
            title_list.pop(-1)
            for i in range(len(title_list) - 1):
                movie_title += title_list[i] + ' '
            movie_title += title_list[len(title_list) - 1]

        # move article to end of title
        articles = ["the", "a", "an"]
        title_words = movie_title.split()
        if title_words[0].lower() in articles:
            movie_title = " ".join(title_words[1:]) + ", " + title_words[0]
        
        ### Title Finding ###
        indices = []
        if year != -1:
            # match only if db title includes specific year
            movie_title += " " + year
            for i in range(len(self.titles)):
                if movie_title == self.titles[i][0].lower():
                    indices.append(i)
        else:
            # match to db title with any year
            for i in range(len(self.titles)):
                db_title_only = re.sub(r'\s*\(\d{4}\)', '', self.titles[i][0])
                if movie_title == db_title_only.lower():
                    indices.append(i)
        return indices

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        stemmed_dict = {} 
        stemmer = PorterStemmer()
        # Stem each word and set the val to what the val was in the sentiment dict
        for word in self.sentiment: 
            stemmed_word = stemmer.stem(word, 0, len(word) - 1)
            stemmed_dict[stemmed_word] = self.sentiment[word]
        
        score = 0 
        # Negation words w/o punc since we remove punc in our input
        negation_words = [
            "not",
            "didnt",
            "never",
            "doesnt",
            "dont",
            "nobody",
            "none",
            "neither",
            "nor"
        ]
        titles = self.extract_titles(preprocessed_input)
        for title in titles: 
            preprocessed_input = preprocessed_input.replace(title, "")
        
        def remove_punctuation(input_string):
            translator = str.maketrans('', '', string.punctuation)
            return input_string.translate(translator)

        preprocessed_input = remove_punctuation(preprocessed_input)
     
        skip_over = False

        words = preprocessed_input.split()

        for i in range(len(words)): 
            if skip_over: 
                skip_over = False
                continue
            word = words[i]
            stemmed_word =  stemmer.stem(word, 0, len(word) - 1)

            # If the word is a negation word and it's not the last word
            if stemmed_word in negation_words and (i < len(words) - 1): 
                following_word = words[i+1]
                following_word = stemmer.stem(following_word, 0, len(following_word) - 1)
                
                # If the following word is pos, subtract 2 since we flip sentiment (e.g. "didn't like")
                # If the following word is neg, add 2 (e.g. "didn't hate")
                if following_word in stemmed_dict and stemmed_dict[following_word] == 'pos':
                    score -= 2
                elif following_word in stemmed_dict and stemmed_dict[following_word] == 'neg': 
                    score += 2
                else: 
                    # Assume for now there are no double negatives 
                    score -= 2
                skip_over = True
            elif stemmed_word in stemmed_dict and stemmed_dict[stemmed_word] == 'pos':
                score += 1
            elif stemmed_word in stemmed_dict and stemmed_dict[stemmed_word] == 'neg':
                score -=1 
        if score < 0: 
            return -1 
        elif score == 0: 
            return 0
        else: 
            return 1 

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.zeros_like(ratings)
        binarized_ratings[(ratings > threshold) & (ratings <= 5)] = 1
        binarized_ratings[(ratings <= threshold) & (ratings > 0)] = -1
        
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        if (np.linalg.norm(u) * np.linalg.norm(v)) == 0:
            return 0
        else:
            similarity = np.dot(u,v) / (np.linalg.norm(u) * np.linalg.norm(v))
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        recommendations = []

        r_xi_list = []
        
        # r_xi_list should be length number of movies
        # for each movie i (row) in the dataset (ratings_matrix)
        for i in range(len(ratings_matrix)):
            # init r_xi for movie i
            r_xi = 0
            # evaluating potential neighbors j
            #  You should use the user_ratings to figure out which movies the user has watched 
            # (these will be your "j" values in the pseudocode above)
            # which movies has the user watched??
            if user_ratings[i] == 0:          
                for j in range(len(ratings_matrix)):
                    # make sure this is not movie i
                    # make sure we have rated this movie before (the corresponding entry in the user_ratings is not 0 or null)
                    if user_ratings[j] != 0:
                        if np.any(ratings_matrix[i]) and np.any(ratings_matrix[j]):
                            vector_i = ratings_matrix[i]
                            vector_j = ratings_matrix[j]
                            # some problems with division by 0 
                            cosine_similarity = self.similarity(vector_i, vector_j)
                            # rating of user x on item j
                            r_xj = user_ratings[j]
                            # update our r_xi value
                            r_xi += (cosine_similarity * r_xj)
                # append the r_xi for movie i into r_xi list
                # the index in r_xi_list corresponds to the movie for that same row in ratings_matrix
            r_xi_list.append(r_xi)
        
        # find the indices of the top k values, this will represent the movie index
        indexed_values = sorted(enumerate(r_xi_list), key=lambda x: x[1], reverse=True)
        recommendations = [index for index, value in indexed_values[0:k]]

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        Your task is to implement the chatbot as detailed in the PA7
        instructions.
        Remember: in the GUS mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')