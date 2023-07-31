from dotenv import load_dotenv, find_dotenv
import os
import openai
import csv
import pandas as pd
import requests
from langchain import OpenAI, LLMChain, PromptTemplate
import streamlit as st
from elevenlabs import generate, play, set_api_key
from datetime import date
from datetime import timedelta

## SETUP ##

load_dotenv(find_dotenv())
openai.api_key=os.getenv("OPENAI_API_KEY")
ELEVEN_LABS_API_KEY=os.getenv("ELEVEN_LABS_API_KEY")
set_api_key(ELEVEN_LABS_API_KEY)

DATE = date.today()
DATE = DATE - timedelta(days = 1)

CSV_NAME = 'news_log.csv'

## IMPORT NEWS ###
def fetch_news(number=10):
    # Fetch tech news from NewsAPI
    url = f"https://newsapi.org/v2/everything?q=(musulman* OR chrétien* OR islam* OR christianisme*)&language=fr&from={DATE}&{DATE}&sortBy=popularity&apiKey={os.getenv('NEWSAPI_API_KEY')}"
    #url = f"https://newsapi.org/v2/top-headlines?country=fr&category=general&apiKey={os.getenv('NEWSAPI_API_KEY')}"
    response = requests.get(url).json()
    print("*********************")
    news_items = response["articles"]
    df = pd.DataFrame(news_items)
    df = df[["title", "description", "url"]].dropna()
    print(df)
    return df.head(number)

#### OpenAI Engine
def openai_request(instructions, task, sample = [], model_engine='gpt-3.5-turbo'):
    prompt = [{"role": "system", "content": instructions }, 
              {"role": "user", "content": task }]
    prompt = sample + prompt
    completion = openai.ChatCompletion.create(model=model_engine, messages=prompt, temperature=0.2, max_tokens=2000)
    return completion.choices[0].message.content


#### Define OpenAI Prompt for news Relevance
def select_relevant_news_prompt(news_articles, topics, n):    
    instructions = f'Your task is to examine a list of News and return a list of {n} boolean values that indicate which of the News are in scope of a list of topics. \
    For each {n} News return a True or False values that indicate the relevance of the News.'
    task =  f"{news_articles} /n {topics}?" 
    sample = [
        {"role": "user", "content": f"[new AI model available from Nvidia, We Exploded the AMD Ryzen 7, Release of b2 Game, XGBoost 3.0 improvices Decision Forest Algorithms, New Zelda Game Now Available, Ukraine Uses a New Weapon] /n [AI, Machine Learning, Data Science, OpenAI, Artificial Intelligence, Data Mining, Data Analytics]?"},
        {"role": "assistant", "content": "[True, False, False, True, False, False]"},
        {"role": "user", "content": f"[Giga Giraff found in Sounth Africa, We Exploded the AMD Ryzen 7, Release of b2 Game] /n [AI, Machine Learning, Data Science, OpenAI, Artificial Intelligence, Data Mining, Data Analytics]?"}, 
        {"role": "assistant", "content": "[False, False, False]"}]
    return instructions, task, sample


#### Define OpenAI Prompt for news Relevance
def check_previous_posts_prompt(title, old_posts):    
    instructions = f'Your objective is to compare a news title with a list of previous news and determine whether it covers a similar topic that was already covered by a previous title. \
        Rate the overlap on a scale between 1 and 10 with 1 beeing a full overlap and 10 representing an unrelated topic. "'
    task =  f"'{title}.' Previous News: {old_posts}."
    sample = [
        {"role": "user", "content": "'Nvidia launches new AI model.' Previous News: [new AI model available from Nvidia, We Exploded the AMD Ryzen 7 7800X3D, The Lara Croft Collection For Switch Has Been Rated By The ESRB]."},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "'Big Explosion of an AMD Ryzen 7.' Previous News: [Improving Mental Wellbeing Through Physical Activity, The Lara Croft Collection For Switch Has Been Rated By The ESRB]."},
        {"role": "assistant", "content": "10"},
        {"role": "user", "content": "'new AI model available from Google.' Previous News : [new AI model available from Nvidia, The Lara Croft Collection For Switch Has Been Rated By The ESRB]."},
        {"role": "assistant", "content": "9"},
        {"role": "user", "content": "'What Really Made Geoffrey Hinton Into an AI Doomer - WIRED.' Previous News : [Why AI's 'godfather' Geoffrey Hinton quit Google, new AI model available from Nvidia, The Lara Croft Collection For Switch Has Been Rated By The ESRB]."},
        {"role": "assistant", "content": "4"}]
    return instructions, task, sample


#### Define OpenAI Prompt for News Tweet
def create_tweet_prompt(title, description):
    #instructions = f'You are a twitter user that creates tweets with a maximum length of 280 characters.'
    instructions = f"Tu es une jeune femme intelligente et drole. Tu travailles dans mon association qui favorise l'entente islamo-chrétienne. Ton travail est de me décrire l'actualité lorsque je te le demande. Tu m'appelles Ali et me parle comme une amie"
    task = f"Tu commences par me saluer et me souhaiter une bonne journée. \
        Comme chaque jour, decris moi brievement l'actualité en te basant des titres et descriptions suivants : \
        Titre: {title}. \
        Description: {description}. \
        Dis si cela peut impacter notre travail dans l'association. Sois optimiste sans trop en faire"
    return instructions, task


#### Define OpenAI Prompt for news Relevance
def previous_post_check(title, old_posts):
    instructions, task, sample = check_previous_posts_prompt(title, old_posts)
    response = openai_request(instructions, task, sample)
    return eval(response)


#### Define OpenAI Prompt for News Tweet
def create_fact_tweet_prompt():
    instructions = f"Tu es une jeune femme intelligente et drole. Tu travailles dans mon association qui favorise l'entente islamo-chrétienne. Ton travail est de me décrire l'actualité lorsque je te le demande. Tu m'appelles Ali et me parle comme une amie"
    task = f"Informe moi qu'il n'y pas d'actualité utile à l'association. Encourage moi. Propose moi de réessayer plus tard. Reste conçis."
    return instructions, task


# Load previous information from a csv file
def get_history_from_csv(csv_name):
    try:
        # try loading the csv file
        df = pd.read_csv(csv_name)
    except:
        # create the csv file
        df = pd.DataFrame(columns=['title'])
        df.to_csv(csv_name, index=False)
    return df

##Functions for Publishing Twitter Tweets
#  # Create the fact tweet
def create_fact_tweet():

    # create a fact tweet
    instructions, tasks = create_fact_tweet_prompt()
    tweet = openai_request(instructions, tasks)

    # tweet creation
    #print(f'Creating fact tweet: {tweet_text}')
        
    # check tweet length and post tweet
    return tweet #Voir si ça marche


def create_news_tweet(title, description):
    # create tiny url
    #tiny_url = create_tiny_url(url)

    # define prompt for tweet creation
    instructions, task = create_tweet_prompt(title, description)
    tweet_text = openai_request(instructions, task)

    #print(f'Creating new tweet: {tweet_text}')
    # check tweet length and post tweet
    with open(f'CSV_NEWS_NAME', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([title])
    return tweet_text
      

#### Main Bot
def main_bot():
    # Read the old CSV data
    # try opening the csv file and creeate it if it does not exist
    df_old_news = get_history_from_csv('CSV_NEWS_NAME')
    df_old_news = df_old_news.tail(16)
    # Fetch news data
    df = fetch_news(15)    
    # Check the Relevance of the News and Filter those not relevant
    relevant_topics ="[Islam, musulman, mosquée, église, chrétien, islamochrétienne, sunnite, chiite, pape, cardinal, monastère, christianisme, laicité, imam, pretre]"
    instructions, task, sample = select_relevant_news_prompt(list(df['title']), relevant_topics, len(list(df['title'])))
    relevance = openai_request(instructions, task, sample)
    print(df)
    print("######")
    relevance_list = eval(relevance)
    print(relevance_list)
    s = 0
    df = df[relevance_list]
    if len(df) > 0:
        #for index, row in df.iterrows():
            #if s == 1:
            #    break
            title = df['title']
            #title = title.replace("'", "")
            description = df['description']
            #description = description.replace("'", "")
                                             
             #if (title not in df_old_news.title.values):
                #doublicate_check = previous_post_check(title, list(df_old_news.tail(10)['title']))
                #if doublicate_check > 3:
                    # Create a tweet
            response = create_news_tweet(title, description)
            return response

                #else: 
                    #print(f"Already tweeted: {title}")
            #else: 
                #reponse = create_fact_tweet()
                #print(reponse)
    else:
        reponse = create_fact_tweet()
        return reponse

#main_bot()

## TEXT TO VOICE ###
def get_voice_message(message):
    audio = generate(
    text=message,
    voice="Nicole",
    model="eleven_multilingual_v1",
    )
    return audio

def main():
    st.set_page_config(
        page_title="Suivre l'actualité - Pour la mission", page_icon=":bird:"
    )
    st.header("Démo projet :bird: ")
    if st.button("Commencer :bird:"):
        st.write("Génération de la réponse...")
        result=main_bot()
        audio = get_voice_message(result)
        st.audio(audio)
        st.info(result)
    


if __name__ == '__main__':
    main()
