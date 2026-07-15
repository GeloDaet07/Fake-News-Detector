import unittest
import json
import warnings
from app import app, model_loaded, transformer_loaded

class TestFakeNewsDetector(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")
        self.app = app.test_client()
        self.app.testing = True

    def test_fake_news_prediction(self):
        if not model_loaded or not transformer_loaded:
            self.skipTest("Models not fully loaded, skipping test.")

        fake_text = """Donald Trump just couldn t wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shout out to his enemies, haters and  the very dishonest fake news media.  The former reality show star had just one job to do and he couldn t do it. As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year,  President Angry Pants tweeted.  2018 will be a great year for America! As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year. 2018 will be a great year for America!  Donald J. Trump (@realDonaldTrump) December 31, 2017Trump s tweet went down about as welll as you d expect.What kind of president sends a New Year s greeting like this despicable, petty, infantile gibberish? Only Trump! His lack of decency won t even allow him to rise above the gutter long enough to wish the American citizens a happy new year!  Bishop Talbert Swan (@TalbertSwan) December 31, 2017no one likes you  Calvin (@calvinstowell) December 31, 2017Your impeachment would make 2018 a great year for America, but I ll also accept regaining control of Congress.  Miranda Yaver (@mirandayaver) December 31, 2017Do you hear yourself talk? When you have to include that many people that hate you you have to wonder? Why do the they all hate me?  Alan Sandoval (@AlanSandoval13) December 31, 2017Who uses the word Haters in a New Years wish??  Marlene (@marlene399) December 31, 2017You can t just say happy new year?  Koren pollitt (@Korencarpenter) December 31, 2017Here s Trump s New Year s Eve tweet from 2016.Happy New Year to all, including to my many enemies and those who have fought me and lost so badly they just don t know what to do. Love!  Donald J. Trump (@realDonaldTrump) December 31, 2016This is nothing new for Trump. He s been doing this for years.Trump has directed messages to his  enemies  and  haters  for New Year s, Easter, Thanksgiving, and the anniversary of 9/11. pic.twitter.com/4FPAe2KypA  Daniel Dale (@ddale8) December 31, 2017Trump s holiday tweets are clearly not presidential.How long did he work at Hallmark before becoming President?  Steven Goodine (@SGoodine) December 31, 2017He s always been like this . . . the only difference is that in the last few years, his filter has been breaking down.  Roy Schulze (@thbthttt) December 31, 2017Who, apart from a teenager uses the term haters?  Wendy (@WendyWhistles) December 31, 2017he s a fucking 5 year old  Who Knows (@rainyday80) December 31, 2017So, to all the people who voted for this a hole thinking he would change once he got into power, you were wrong! 70-year-old men don t change and now he s a year older.Photo by Andrew Burton/Getty Images."""
        
        response = self.app.post('/predict', data=json.dumps({'text': fake_text}), content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['prediction'], "Fake News")
        self.assertFalse(data['mock'])

    def test_real_news_prediction(self):
        if not model_loaded or not transformer_loaded:
            self.skipTest("Models not fully loaded, skipping test.")

        real_text = """two senior republican u.s. senators criticized secretary of state rex tillerson on sunday for saying that russia may have the right approach on syria and for what they called his lack of focus on afghanistan and pakistan. his statements about syria really disturb me. no, russian president vladimir putin does not have it right when it comes to syria, senator lindsey graham said. in separate television interviews, graham and senator john mccain, prominent republican foreign policy voices, took aim at tillerson s remarks last week that russia may have got the right approach and the united states the wrong approach to syria. russia has backed president bashar al-assad in syria s civil war, while the united states supports rebel groups trying to overthrow him. mccain told cbs face the nation that he sometimes regretted backing tillerson s nomination by republican president donald trump and that his comments on russia being right on syria made him emotional and upset. i know what the slaughter has been like. i know that the russians knew that bashar assad was going to use chemical weapons. and to say that maybe we ve got the wrong approach? he said. both senators backed the nomination of tillerson in january, even while expressing concern about his dealings with russia when he was chief executive of exxonmobil. xom.n graham, who visited afghanistan and pakistan last week with mccain, accused tillerson of being awol on the two countries and failing to fill key state department posts. i am so worried about the state department, graham said on nbc s meet the press. a state department official responded to the criticism of tillerson by saying that a u.s.-russian-brokered ceasefire for southwest syria was an example of what the secretary had described as the potential to coordinate with russia, in spite of unresolved differences, to produce stability and serve our mutual security interests. the official, who did not want to be identified, also said the state department was taking an active role in a review of afghanistan and pakistan policy and continued to work with the white house on nominations. since the exit of most foreign troops in 2014, afghanistan s u.s.-backed government has lost ground to a taliban insurgency in a war that kills and maims thousands of civilians each year."""

        response = self.app.post('/predict', data=json.dumps({'text': real_text}), content_type='application/json')
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['prediction'], "Real News")
        self.assertFalse(data['mock'])

import time
import sys

class JestLikeTestResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.start_times = {}

    def startTest(self, test):
        self.start_times[test] = time.time()
        super().startTest(test)

    def addSuccess(self, test):
        super().addSuccess(test)
        elapsed = (time.time() - self.start_times[test]) * 1000
        print(f"  \033[32m✓\033[0m \033[90m{test._testMethodName}\033[0m ({elapsed:.0f} ms)")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        elapsed = (time.time() - self.start_times[test]) * 1000
        print(f"  \033[31m✕\033[0m \033[90m{test._testMethodName}\033[0m ({elapsed:.0f} ms)")

    def addError(self, test, err):
        super().addError(test, err)
        elapsed = (time.time() - self.start_times[test]) * 1000
        print(f"  \033[31m✕\033[0m \033[90m{test._testMethodName}\033[0m ({elapsed:.0f} ms)")
        
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        print(f"  \033[33m○\033[0m \033[90m{test._testMethodName}\033[0m (skipped)")

class JestLikeTestRunner:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout

    def run(self, test):
        result = JestLikeTestResult()
        test(result)
        return result

if __name__ == '__main__':
    print("\n\033[43;30m RUNS \033[0m \033[1mtest_app.py\033[0m\n")
    
    start_time = time.time()
    # We load the tests manually instead of using unittest.main to fully control the runner
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFakeNewsDetector)
    runner = JestLikeTestRunner()
    result = runner.run(suite)
    elapsed = time.time() - start_time
    
    passed = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
    
    print()
    if result.wasSuccessful():
        print(f"\033[42;30m PASS \033[0m \033[1mtest_app.py\033[0m")
    else:
        print(f"\033[41;30m FAIL \033[0m \033[1mtest_app.py\033[0m")
    
    print(f"Tests:       \033[32m{passed} passed\033[0m, {result.testsRun} total")
    print(f"Time:        {elapsed:.3f} s")
    print()
    
    if not result.wasSuccessful():
        sys.exit(1)
