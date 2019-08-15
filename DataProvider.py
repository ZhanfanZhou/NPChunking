class TweetWarehouse:
    def __init__(self):
        self.id2tweet = {}
        self.id2label = {}

    def put_in(self, id, tweet, label):
        self.id2tweet[id] = tweet
        self.id2label[id] = label

    def get_tweet(self, id):
        if self.id2tweet:
            return self.id2tweet.get(id, "TWEET NOT FOUND ERROR")
        else:
            print("TweetWarehouse not initialized!")

    def get_label(self, id):
        if self.id2label:
            return self.id2label.get(id, "TWEET NOT FOUND ERROR")
        else:
            print("TweetWarehouse not initialized!")
