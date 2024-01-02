# indian-sign-language

After watching countless videos on Youtube on "how to build a sign language interpreter" I found a few problems. The biggest one was that almost every solution used images instead of landmarks, not accounting for changes in lighting, distance from camera and skin-tones for the end users.

This project aims to solve that problem by using Mediapipe's landmarks feature.



I'm estimating the distance of each landmark of the hand from the center of the user's wrist.

After normalizing that distance, I trained a simple model with sci-kit-learn in test.py.
![AYAAN_SignLang](https://github.com/ayaanjamil/indian-sign-language/assets/39400870/ff1de169-6fa5-4559-9b6c-404ed1f2925c)


You can train a similar model by:

1. running csv-write.py (make signs for each alphabet, essentially capturing your training data)

2. running merge-data.py (~~call it lack of foresight but this~~ converts the csv data into our training data)

3. train our model with train.py

4. run the trained model with pred.py



Future scope:

As this can work only with alphabets now, I think it can be improved by incorporating Long Short-Term Memory ([LTSM]([Long short-term memory - Wikipedia](https://en.wikipedia.org/wiki/Long_short-term_memory))) to add gesture support so users can interpret full sentences.

Other than that, I think by using Mediapipe's holistic model, instead of Hands model, multi-hand gestures, along with facial gestures can be incorporated.

