import pandas as pd
import numpy as np
import math
import nltk
from nltk.corpus import stopwords
import pymorphy3
from functools import lru_cache # для кэширования лемм

nltk.download('stopwords', quiet=True)
russian_stopwords = set(stopwords.words("russian")) - set(['не', 'нет', 'хорошо', 'есть'])
morph = pymorphy3.MorphAnalyzer()

def Create_join_tables(df_bookings, df_reviews):
    df_bookings['booking_date'] = pd.to_datetime(df_bookings['booking_date'])
    df_bookings['returning_customer'] = df_bookings['returning_customer'].astype('int')
    df_bookings['booking_status'] = (df_bookings['booking_status'] == 'отказ_брони').astype('int')
    df_bookings.loc[df_bookings['meal_plan'] == 'не выбрант', 'meal_plan'] = 'не выбран'
    df_bookings['parking_included'] = df_bookings['parking_included'].astype('int')
    df_bookings.loc[df_bookings['adult_count'] > 99, 'adult_count'] = df_bookings.loc[df_bookings['adult_count'] > 99, 'adult_count']/100
    df_reviews['review_date'] = pd.to_datetime(df_reviews['review_date'])
    df_bookings = df_bookings.drop_duplicates(subset=['booking_id', 'booking_date', 'booking_value', 'booking_status'],
                                              keep='first', ignore_index=True)
    df_bookings = df_bookings[df_bookings['booking_value'] > 0].copy().reset_index(drop=True)

    reviews_by_customer = df_reviews.groupby(['customer_id', 'review_date'], as_index=False).agg( 
        stay_rating=('stay_rating', 'mean'),
        review_text=('review_text', lambda s: ' '.join(s.astype(str))))

    review_pairs = df_reviews[['customer_id', 'booking_id']].drop_duplicates(subset='booking_id')

    bookings_with_customer = df_bookings.merge( review_pairs, on='booking_id', how='left')

    reviews_by_customer = reviews_by_customer.sort_values('review_date').reset_index(drop=True)
    bookings_with_customer = bookings_with_customer.sort_values('booking_date').reset_index(drop=True)

    result = pd.merge_asof(bookings_with_customer, reviews_by_customer, left_on='booking_date', right_on='review_date',
                           by='customer_id', direction='backward', allow_exact_matches=False)
    result['review_text'] = result['review_text'].fillna('ААААААтсутствуетОтзывml')

    result = result.rename(columns={'review_date': 'client_previous_review_date',
                                    'stay_rating': 'client_previous_rating',
                                    'review_text': 'client_previous_review'})


    reviews_by_date = df_reviews.groupby('review_date', as_index=False).agg(
        stay_rating=('stay_rating', 'mean'),
        review_text=('review_text', lambda s: ' '.join(s.astype(str))))


    reviews_by_date = reviews_by_date.sort_values('review_date').reset_index(drop=True)
    result = result.sort_values('booking_date').reset_index(drop=True)

    df = pd.merge_asof(result, reviews_by_date, left_on='booking_date', right_on='review_date', 
                       direction='backward', allow_exact_matches=False)

    df = df.rename(columns={'review_date': 'last_review_date', 'stay_rating': 'previous_rating', 'review_text': 'previous_review'})
    df['previous_review'] = df['previous_review'].fillna('ААААААтсутствуетОтзывml')

    df['booking_month'] = df['booking_date'].dt.month
    df['checkin_month'] = (df['booking_date'] + pd.to_timedelta(df['days_until_checkin'], unit='D')).dt.month

    df['booking_month_sn'] = np.sin(df['booking_month'] * math.pi / 6)
    df['booking_month_cs'] = np.cos(df['booking_month'] * math.pi / 6)
    df['checkin_month_sn'] = np.sin(df['checkin_month'] * math.pi / 6)
    df['checkin_month_cs'] = np.cos(df['checkin_month'] * math.pi / 6)

    df['total_nights'] = df['weekend_nights'] + df['weekday_nights']
    df['weekend_share'] = df['weekend_nights'] / df['total_nights']
    df['weekday_share'] = df['weekday_nights'] / df['total_nights']
    df['price_per_night'] = df['booking_value'] / df['total_nights']
    df['total_guests'] = df['adult_count'] + df['child_count']
    df['price_per_guest'] = df['booking_value'] / df['total_guests']

    for col in ['client_previous_review', 'previous_review']:
        s = df[col].str.replace(r'\W+', ' ', regex=True).str.strip().str.lower()
        unique_texts = s.unique()

        processed = { text:preprocess_text(text.split()) for text in unique_texts}
        df[col] = s.map(processed)

    result = df.drop(columns = ['booking_id', 'customer_id', 'client_previous_review_date',
                                'last_review_date', 'booking_month', 'checkin_month'])
    for col in ['sales_channel', 'meal_plan', 'room_type']:
        result[col] = result[col].astype('category')

    return result

@lru_cache(maxsize=100_000)
def lemmatize_word(word):
    return morph.parse(word)[0].normal_form


def preprocess_text(words_list):
    return " ".join(lemmatize_word(word) for word in words_list if word not in russian_stopwords)
