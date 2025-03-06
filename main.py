import os
import json
import re
import random
from typing import Dict, List, Tuple
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Initialize Flask application
app = Flask(__name__)

class GossipGirlFinanceBot:
    def __init__(self, knowledge_base_path: str = "financial_knowledge.json"):
        """
        Initialize the Gossip Girl-themed Financial Chatbot
        
        Args:
            knowledge_base_path: Path to the JSON file containing financial information
        """
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        
        # Initialize the NLP model for question answering
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
        self.qa_pipeline = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer)
        
        # Initialize greeting and farewell phrases
        self.greeting_phrases = ["hello", "hi", "hey", "greetings", "howdy"]
        self.farewell_phrases = ["bye", "goodbye", "exit", "quit", "see you"]
        
        # Gossip Girl intros
        self.gossip_intros = [
            "Hello Upper East Siders. Gossip Girl here, your one and only financial source into the scandalous lives of Manhattan's elite. ",
            "Spotted: Your favorite financial insider with some juicy money tea. ",
            "Hey there, financial wannabes. Ready for today's most exclusive fiscal dirt? ",
            "Good morning, Money Mavens. Word on the street is someone's asking about their finances. ",
            "Breaking news from the financial district, and you heard it here first. ",
            "Attention Upper East Siders, I have the ultimate scoop on what's happening in the money world. "
        ]
        
        # Gossip Girl outros
        self.gossip_outros = [
            " You know you love me. XOXO, Financial Girl.",
            " And who am I? That's one secret I'll never tell. You know you love my financial advice.",
            " Spotted: You, making smarter money moves after this little chat.",
            " Whether you're old money or new money, now you're in the know.",
            " And that's the kind of wealth management that keeps you on the social register.",
            " Until next time, keep your friends close and your investments closer."
        ]
        
        # Add normal definitions
        self.normal_definitions = {

            "mutual fund": "A mutual fund is a pool of money collected from multiple investors that is professionally managed and invested in various securities like stocks, bonds, and other assets. The fund manager makes investment decisions on behalf of all investors, who share in the profits, losses, and expenses proportionally.",
            "stock": "A stock represents partial ownership in a company, giving shareholders a claim on its assets and earnings. Stocks are bought and sold on stock exchanges, and their value fluctuates based on company performance and market conditions.",
            "bond": "A bond is a fixed-income investment where an investor loans money to a government or corporation for a set period in exchange for periodic interest payments and the return of the principal amount at maturity.",
            "401k": "A 401(k) is a retirement savings plan sponsored by an employer, allowing employees to contribute pre-tax income, which grows tax-deferred until withdrawal, typically after retirement.",
            "ira": "An Individual Retirement Account (IRA) is a tax-advantaged account that helps individuals save for retirement. Traditional IRAs allow tax-deductible contributions, while Roth IRAs offer tax-free withdrawals.",
            "etf": "An Exchange-Traded Fund (ETF) is an investment fund that holds a collection of securities, such as stocks or bonds, and trades on stock exchanges like a stock. ETFs provide diversification and lower expense ratios compared to mutual funds.",
            "inflation": "Inflation is the rate at which the general level of prices for goods and services rises over time, reducing the purchasing power of money.",
            "compound interest": "Compound interest is the process where interest is added to the initial principal amount, and future interest is earned on both the principal and the accumulated interest, leading to exponential growth over time.",
            "diversification": "Diversification is an investment strategy that involves spreading investments across different asset classes to reduce risk. A well-diversified portfolio minimizes potential losses by avoiding overconcentration in a single asset.",
            "credit score": "A credit score is a numerical representation of a person’s creditworthiness, based on their credit history, debt levels, and payment behavior. It affects loan approvals, interest rates, and financial opportunities.",
            "roth ira": "A Roth IRA is a retirement savings account where contributions are made with after-tax income, allowing tax-free withdrawals in retirement, provided certain conditions are met.",
            "bull market": "A bull market refers to a prolonged period of rising stock prices, often driven by strong economic conditions, investor confidence, and increasing corporate profits.",
            "bear market": "A bear market is a period when stock prices decline by at least 20% from recent highs, often due to economic downturns, declining investor confidence, or external financial shocks.",
            "liquidity": "Liquidity refers to how easily an asset can be converted into cash without significantly affecting its price. Cash is the most liquid asset, while real estate and certain investments are less liquid.",
            "dividend": "A dividend is a portion of a company's earnings distributed to shareholders, usually in cash or additional shares, as a reward for investing in the company.",
            "portfolio": "A portfolio is a collection of financial assets, such as stocks, bonds, mutual funds, and real estate, that an investor owns. A well-balanced portfolio is key to managing risk and achieving financial goals.",
            "hedge fund": "A hedge fund is an alternative investment vehicle that pools capital from accredited investors to employ various strategies, such as long-short positions and derivatives, to maximize returns while managing risk.",
            "private equity": "Private equity refers to investments in privately held companies or buyouts of publicly traded companies, often involving direct investment strategies and active management to improve financial performance.",
            "venture capital": "Venture capital is a form of private equity financing provided to startups and early-stage companies with high growth potential in exchange for equity ownership.",
            "asset allocation": "Asset allocation is the process of dividing an investment portfolio among different asset categories, such as stocks, bonds, and cash, to balance risk and reward according to an investor’s goals and risk tolerance.",
            "market capitalization": "Market capitalization (market cap) is the total value of a company’s outstanding shares, calculated by multiplying the stock price by the number of shares outstanding. It indicates a company's size and market value.",
            "dollar cost averaging": "Dollar-cost averaging is an investment strategy where an investor regularly invests a fixed amount of money into a particular asset, regardless of its price, reducing the impact of market fluctuations over time.",
            "index fund": "An index fund is a type of mutual fund or ETF designed to track the performance of a specific market index, such as the S&P 500. It provides broad market exposure and low costs.",
            "rebalancing": "Rebalancing is the process of adjusting the allocation of assets in an investment portfolio to maintain the desired level of risk and return as market conditions change.",
            "capital gain": "A capital gain is the profit earned when an asset, such as a stock or real estate, is sold for more than its purchase price. Long-term capital gains often receive favorable tax treatment.",
            "capital loss": "A capital loss occurs when an asset is sold for less than its purchase price. Capital losses can offset capital gains for tax purposes, reducing taxable income.",
            "tax-loss harvesting": "Tax-loss harvesting is a strategy where investors sell securities at a loss to offset capital gains, reducing their overall tax liability while maintaining an investment strategy.",
            "emergency fund": "An emergency fund is a reserve of liquid assets set aside to cover unexpected financial expenses, such as medical emergencies or job loss, providing financial security and stability.",
            "robo-advisor": "A robo-advisor is an automated platform that provides investment management services using algorithms to create and manage portfolios based on an investor's risk tolerance and goals.",
            "yield": "Yield refers to the income generated from an investment, typically expressed as a percentage. It includes interest from bonds and dividends from stocks, indicating an investment’s profitability."
 
        }
        
        print("Gossip Girl Financial Bot initialized! Ready to spill the tea on money matters.")
    
    def _load_knowledge_base(self, file_path: str) -> Dict:
        """Load the financial knowledge base from a JSON file"""
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            # Create default knowledge base with Gossip Girl style
            default_kb = {

                "mutual fund": "A mutual fund is the ultimate Upper East Side investment clique, darling. Investors pool their money, and a financial Chuck Bass makes all the decisions—stocks, bonds, the whole portfolio. Everyone shares in the wins, the losses, and of course, the management fees. Because even in finance, nothing comes for free.",
                "stock": "A stock is like holding a VIP pass to a company’s success—own a share, own a piece. The more you have, the more influence you wield, just like Blair Waldorf at Constance. Play it right, and your investment climbs faster than Serena’s social status. But beware, stocks can fall just as fast as they rise.",
                "bond": "A bond is like lending cash to a government or corporation with the promise of getting paid back—with interest, of course. Less risky than stocks, but also less thrilling—think Nate Archibald over Chuck Bass. Old money adores bonds because they value stability over scandal.",
                "401k": "A 401(k) is the trust fund you actually have to build yourself. Your employer sets it up, you contribute pre-tax dollars, and it grows until retirement—tax-free, of course. Think of it as securing your future penthouse, because even the elite plan ahead.",
                "ira": "An IRA is your personal financial safety net, no trust fund required. Choose Traditional to delay taxes, or Roth to pay now and withdraw tax-free later. Either way, it’s like choosing between drama now or drama later—both have their perks.",
                "etf": "An ETF is like a front-row seat to the stock market’s best players. It’s a mix of investments traded throughout the day—unlike mutual funds, which only trade once. Instant diversification, zero commitment. Sounds a lot like Serena’s dating strategy.",
                "inflation": "Inflation is the reason your money buys less each year—think of it as designer prices creeping up season after season. Central banks fight it by raising interest rates, but too much control? That’s a fashion disaster waiting to happen.",
                "compound interest": "Compound interest is the ultimate financial glow-up—earn interest on your interest, and watch your money multiply faster than Gossip Girl rumors. The longer you let it work, the bigger the payoff. Patience, darling, is a wealth-building virtue.",
                "diversification": "Diversification is the golden rule of investing—never put all your social capital, or money, in one place. Stocks, bonds, real estate—mix it up. If one crashes, the others keep you afloat. The old-money elite have been doing this for generations.",
                "credit score": "Your credit score is your financial reputation, darling. One bad move—missed payments, too much debt—and it haunts you like a Gossip Girl blast. Keep it high, and doors (and exclusive credit cards) open effortlessly.",
                "roth ira": "A Roth IRA is like prepaying for luxury now so you can enjoy it tax-free later. You contribute already-taxed income, but future withdrawals? Totally untaxed. The Upper East Side calls that smart planning. New money should take notes.",
                "bull market": "A bull market is the Wall Street version of Fashion Week—everyone’s thriving, stocks are soaring, and fortunes are multiplying. But like any party, it won’t last forever. The smartest investors know when to step out before the crash.",
                "bear market": "A bear market is when stock prices drop at least 20%, and suddenly, everyone’s panicking. Think of it as social exile—reputations (and portfolios) take a hit, but the strong always make a comeback. The question is: are you patient enough to wait?",
                "liquidity": "Liquidity is how quickly you can turn assets into cash—because sometimes, you need an emergency shopping spree at Bergdorf’s. Cash? Instantly liquid. A Hamptons mansion? Not so much. The truly wealthy keep a balance of both.",
                "dividend": "A dividend is passive income at its finest—companies sharing profits with shareholders like an elite allowance. Old money loves them because they don’t have to lift a finger. Blair Waldorf would definitely approve.",
                "portfolio": "Your portfolio is your financial wardrobe—diverse, strategic, and tailored to your goals. Stocks for drama, bonds for stability, maybe some real estate for flair. The key? Balance. Even the most fashionable icons mix classic with trendy.",
                "hedge fund": "A hedge fund is the VIP after-party of investing—exclusive, high-stakes, and only for the wealthy. They use complex strategies to win big, no matter the market. Risky? Absolutely. But the elite never play it safe.",
                "private equity": "Private equity is next-level investing—buying entire companies instead of just shares. It’s like acquiring a whole fashion empire instead of just a designer handbag. Requires serious money, but the returns? Très chic.",
                "venture capital": "Venture capital is betting on the next big thing before it’s cool—funding startups in hopes of discovering the next Uber or Instagram. High risk, but if it pays off? You’re looking at generational wealth. Very Bass Industries.",
                "asset allocation": "Asset allocation is like curating the perfect guest list—balance is key. Stocks for excitement, bonds for stability, and cash for security. The mix depends on your risk tolerance—are you a safe Lily or a bold Chuck?",
                "market capitalization": "Market cap ranks companies like social status on the Upper East Side. Large caps are the Blairs and Serenas—established, powerful. Mid caps? Nates and Chucks—rising stars. Small caps? Jenny Humphreys—high risk, high reward.",
                "dollar cost averaging": "Dollar cost averaging is playing the long game—investing steadily instead of all at once. It smooths out the highs and lows, so you’re not caught buying at the worst time. Think of it as effortless, drama-free investing.",
                "index fund": "An index fund is the ultimate set-it-and-forget-it investment—like letting Dorota handle your social calendar. It tracks the market automatically, has low fees, and delivers solid returns. Even Warren Buffett approves.",
                "rebalancing": "Rebalancing is the key to maintaining power—er, wealth. Over time, some investments overperform while others lag. Adjusting your portfolio keeps it in check. Even the elite refine their strategies regularly.",
                "capital gain": "A capital gain is making money off an investment—buy low, sell high, cash in. Hold for over a year, and you get tax perks. The truly wealthy play the long game, just like in high society.",
                "capital loss": "A capital loss is selling an investment for less than you paid—financial heartbreak, but sometimes useful. You can use it to lower your tax bill. Even a scandal can be spun into something beneficial.",
                "tax-loss harvesting": "Tax-loss harvesting is using one financial loss to offset another—think of it as damage control. Sell underperforming assets, claim the loss, and reinvest smartly. Even the IRS allows a well-executed redemption arc.",
                "emergency fund": "An emergency fund is your financial safety net—cash reserves to avoid selling investments or taking on debt when life happens. Old money keeps them, new money forgets. Be old money, darling.",
                "robo-advisor": "A robo-advisor is like having an algorithm for a financial planner—no emotions, no drama, just automated investing based on your goals. Perfect for those who prefer efficiency over human error.",
                "yield": "Yield is your investment’s performance score—how much you’re earning from dividends or interest. Higher yield? More rewards, but often more risk. The key is knowing when to go big and when to play it safe."

            }
            
            # Save the knowledge base to file
            with open(file_path, 'w') as file:
                json.dump(default_kb, file, indent=4)
            
            return default_kb
    
    def _save_knowledge_base(self, file_path: str) -> None:
        """Save the current knowledge base to the specified file path"""
        with open(file_path, 'w') as file:
            json.dump(self.knowledge_base, file, indent=4)
    
    def _find_best_match(self, query: str) -> Tuple[str, float]:
        """Find the best matching topic for the given query"""
        query = query.lower()
        best_match = None
        highest_score = 0
        
        # Simple keyword matching
        for topic, description in self.knowledge_base.items():
            if topic in query:
                # If exact topic name is in the query, that's a strong match
                return topic, 0.9
            
            # Calculate word overlap between query and topic
            topic_words = set(re.findall(r'\w+', topic))
            query_words = set(re.findall(r'\w+', query))
            common_words = topic_words.intersection(query_words)
            
            if common_words:
                score = len(common_words) / len(topic_words)
                if score > highest_score:
                    highest_score = score
                    best_match = topic
        
        return best_match, highest_score
    
    def _is_greeting(self, message: str) -> bool:
        """Check if the message is a greeting"""
        return any(phrase in message.lower() for phrase in self.greeting_phrases)
    
    def _is_farewell(self, message: str) -> bool:
        """Check if the message is a farewell"""
        return any(phrase in message.lower() for phrase in self.farewell_phrases)
    
    def _generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer for the query based on the context using the QA model
        For simple terminology queries, we'll just use the Gossip Girl styled context
        """
        if len(context.split()) < 100:  # For short contexts, just return as is
            return context
            
        try:
            result = self.qa_pipeline(question=query, context=context)
            if result['score'] < 0.5:
                return context
            return result['answer']
        except:
            return context
    
    def _process_learning_request(self, message: str) -> str:
        """Process a request to teach the chatbot new information"""
        pattern = r"(?:learn|add|teach) (?:that|about) ([a-z0-9 ]+) (?:is|are|means) (.+)"
        match = re.search(pattern, message.lower())
        
        if match:
            topic = match.group(1).strip()
            definition = match.group(2).strip()
            
            # Add Gossip Girl flair to the definition
            gossip_definition = f"{definition} And that's a financial secret even I didn't know until now. The elite of Manhattan would pay good money for this kind of insider knowledge."
            
            # Add to knowledge base
            self.knowledge_base[topic] = gossip_definition
            self._save_knowledge_base("financial_knowledge.json")
            
            return f"Spotted: New financial intel entering my database. {topic} is {definition} XOXO, you know you love teaching me."
        
        return "Even Gossip Girl needs clear information. Try using the format: 'Learn that [topic] is [definition]'"
    
    def get_all_topics(self) -> List[str]:
        """Return all available financial topics"""
        return sorted(list(self.knowledge_base.keys()))
    
    def add_gossip_girl_flair(self, response: str) -> str:
        """Add Gossip Girl style to a response"""
        # Add intro and outro, but keep the main content intact
        intro = random.choice(self.gossip_intros)
        outro = random.choice(self.gossip_outros)
        final_response = f"{intro}{response}{outro}"
        return final_response
    
    def respond(self, message: str, style: str = "gossip") -> str:
        """
        Generate a response for the user's query
        
        Args:
            message: The user's message
            style: The style of response ("gossip" or "normal")
        """
        # Check for special commands
        if self._is_greeting(message):
            greeting_response = "Hello there! I'm your exclusive source into the scandalous lives of financial terms. What money gossip can I spill today?"
            return self.add_gossip_girl_flair(greeting_response) if style == "gossip" else greeting_response
        
        if self._is_farewell(message):
            return "You know you'll miss me. Until next time, XOXO, Financial Girl." if style == "gossip" else "Goodbye! Feel free to come back with more financial questions."
        
        # Check if the user is trying to teach the chatbot
        if re.search(r"(?:learn|add|teach)", message.lower()):
            return self._process_learning_request(message)
        
        # Find best matching topic
        best_match, confidence = self._find_best_match(message)
        
        if best_match and confidence > 0.3:
            # Get the appropriate definition based on style
            if style == "normal" and best_match in self.normal_definitions:
                return self.normal_definitions[best_match]
            elif style == "gossip":
                context = self.knowledge_base[best_match]
                answer = self._generate_answer(message, context)
                return self.add_gossip_girl_flair(answer)
            else:
                # Fallback to gossip style if normal isn't available
                context = self.knowledge_base[best_match]
                answer = self._generate_answer(message, context)
                return self.add_gossip_girl_flair(answer) if style == "gossip" else answer
        
        # Default response
        if style == "gossip":
            default_responses = [
                "Even Gossip Girl doesn't have all the financial tea on that. Try asking about specific terms like 'robo-advisor' or 'yield'. I promise the scandal is worth it.",
                "That's not in my financial diary yet. I'm more versed in terms like 'robo-advisor' or 'yield'. Ask me about those instead, and I'll spill all the details.",
                "That financial query is more mysterious than Gossip Girl's identity. Try something like 'What is yield?' or 'Tell me about robo-advisors' instead.",
                "That's not trending in my financial circles yet. But I have plenty of gossip on 'robo-advisors' or 'yield' if you're interested."
            ]
            return random.choice(default_responses)
        else:
            return "I don't have information on that topic. Try asking about 'robo-advisor' or 'yield' instead."

# Initialize the chatbot
chatbot = GossipGirlFinanceBot()

@app.route('/')
def home():
    """Render the home page"""
    topics = chatbot.get_all_topics()
    return render_template('index.html', topics=topics)

@app.route('/ask', methods=['POST'])
def ask():
    """Process user questions"""
    data = request.get_json()
    user_message = data.get('message', '')
    style = data.get('style', 'gossip')  # Default to gossip style
    
    if not user_message:
        return jsonify({'response': 'Please enter a question, darling.'})
    
    response = chatbot.respond(user_message, style)
    return jsonify({'response': response})

if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create the index.html template file with Gossip Girl style
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gossip Girl Financial - XOXO</title>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@300;400;700&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Lato', sans-serif;
            background-color: #f5f5f5;
            background-image: url('https://www.transparenttextures.com/patterns/subtle-white-feathers.png');
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Playfair Display', serif;
        }
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 90vh;
            border: 1px solid #e0e0e0;
        }
        .chat-header {
            background-color: #1b1b1b;
            color: #d4af37;
            padding: 15px;
            text-align: center;
            border-bottom: 3px solid #d4af37;
        }
        .chat-header p {
            color: #f8f8f8;
        }
        .messages-area {
            flex-grow: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            background-image: url('https://www.transparenttextures.com/patterns/cubes.png');
            background-color: #f9f9f9;
        }
        .message {
            max-width: 75%;
            padding: 10px 15px;
            border-radius: 20px;
            margin-bottom: 12px;
            animation: fadeIn 0.3s;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #e9e9e9;
            align-self: flex-end;
            border-bottom-right-radius: 5px;
            border-left: 3px solid #6c757d;
        }
        .bot-message {
            background-color: #f8f3e3;
            align-self: flex-start;
            border-bottom-left-radius: 5px;
            border-right: 3px solid #d4af37;
            font-style: italic;
            color: #333;
        }
        .normal-message {
            background-color: #e8f4f8;
            align-self: flex-start;
            border-bottom-left-radius: 5px;
            border-right: 3px solid #4682B4;
            font-style: normal;
            color: #333;
        }
        .input-area {
            padding: 15px;
            background-color: #1b1b1b;
            border-top: 1px solid #d4af37;
        }
        .topics-sidebar {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #d4af37;
            border-radius: 5px;
            margin-top: 15px;
            background-color: #f8f3e3;
        }
        .topic-item {
            cursor: pointer;
            padding: 8px 15px;
            transition: background-color 0.2s;
            border-bottom: 1px solid #eee;
            font-family: 'Playfair Display', serif;
        }
        .topic-item:hover {
            background-color: #d4af37;
            color: white;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .typing-indicator {
            padding: 10px 15px;
        }
        .typing-indicator span {
            height: 8px;
            width: 8px;
            background-color: #d4af37;
            border-radius: 50%;
            display: inline-block;
            margin: 0 1px;
            animation: typing 1s infinite;
        }
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }
        @keyframes typing {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        .btn-gold {
            background-color: #d4af37;
            border-color: #d4af37;
            color: #1b1b1b;
        }
        .btn-gold:hover {
            background-color: #c19d2a;
            border-color: #c19d2a;
            color: #fff;
        }
        .card-header {
            font-family: 'Playfair Display', serif;
            letter-spacing: 1px;
        }
        .form-control {
            border: 1px solid #d4af37;
            border-radius: 20px;
            background-color: #f8f8f8;
        }
        .form-control:focus {
            border-color: #d4af37;
            box-shadow: 0 0 0 0.2rem rgba(212, 175, 55, 0.25);
        }
        .xoxo {
            font-family: 'Playfair Display', serif;
            font-style: italic;
            color: #d4af37;
        }
        .chat-bubble-tail {
            height: 15px;
            width: 15px;
            position: relative;
        }
        /* Toggle Switch Styles */
        .toggle-switch-container {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 8px 15px;
            margin-bottom: 10px;
            text-align: center;
            background-color: #f8f8f8;
            border-bottom: 1px solid #e0e0e0;
        }
        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 30px;
            margin: 0 10px;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #e8f4f8;
            transition: .4s;
            border-radius: 34px;
            border: 1px solid #4682B4;
        }
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 22px;
            width: 22px;
            left: 4px;
            bottom: 3px;
            background-color: #4682B4;
            transition: .4s;
            border-radius: 50%;
        }
        input:checked + .toggle-slider {
            background-color: #f8f3e3;
            border: 1px solid #d4af37;
        }
        input:checked + .toggle-slider:before {
            transform: translateX(29px);
            background-color: #d4af37;
        }
        .toggle-label {
            font-family: 'Playfair Display', serif;
            font-weight: 600;
            font-size: 14px;
        }
        .normal-label {
            color: #4682B4;
        }
        .gossip-label {
            color: #d4af37;
        }
    </style>
</head>
<body>
    <div class="container py-4">
        <div class="chat-container">
            <div class="chat-header">
                <h1>Gossip Girl Financial <span class="xoxo">XOXO</span></h1>
                <p>Your one and only source into the scandalous lives of financial terms.</p>
            </div>
            
            <div class="toggle-switch-container">
                <span class="toggle-label normal-label">Normal</span>
                <label class="toggle-switch">
                    <input type="checkbox" id="styleToggle" checked>
                    <span class="toggle-slider"></span>
                </label>
                <span class="toggle-label gossip-label">Gossip Girl</span>
            </div>
            
            <div class="messages-area" id="chat-messages">
                <!-- Messages will appear here -->
                <div class="message bot-message">
                    Hello Upper East Siders. Gossip Girl here, your one and only financial source into the scandalous lives of Manhattan's elite. What financial dirt can I dish out for you today? You know you love me. XOXO, Financial Girl.
                </div>
            </div>
            <div class="input-area">
                <div class="input-group">
                    <input type="text" id="user-input" class="form-control" placeholder="Ask me about financial terms...">
                    <button class="btn btn-gold" id="send-button">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-send" viewBox="0 0 16 16">
                            <path d="M15.854.146a.5.5 0 0 1 .11.54l-5.819 14.547a.75.75 0 0 1-1.329.124l-3.178-4.995L.643 7.184a.75.75 0 0 1 .124-1.33L15.314.037a.5.5 0 0 1 .54.11ZM6.636 10.07l2.761 4.338L14.13 2.576 6.636 10.07Zm6.787-8.201L1.591 6.602l4.339 2.76 7.494-7.493Z"/>
                        </svg>
                    </button>
                </div>
            </div>
        </div>
        
        <div class="card mt-4">
            <div class="card-header bg-dark text-gold">
                <h5 class="mb-0 text-white">Financial Topics <span class="xoxo">XOXO</span></h5>
            </div>
            <div class="card-body topics-sidebar">
                <div class="row">
                    {% for topic in topics %}
                    <div class="col-md-4 col-sm-6">
                        <div class="topic-item" onclick="askAbout('{{ topic }}')">{{ topic }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        $(document).ready(function() {
            // Function to add a message to the chat
            function addMessage(message, isUser, style) {
                const messagesArea = $('#chat-messages');
                let messageClass = isUser ? 'user-message' : (style === 'normal' ? 'normal-message' : 'bot-message');
                const messageDiv = $('<div class="message ' + messageClass + '"></div>');
                messageDiv.text(message);
                messagesArea.append(messageDiv);
                // Scroll to the bottom
                messagesArea.scrollTop(messagesArea[0].scrollHeight);
            }
            
            // Function to show typing indicator
            function showTypingIndicator() {
                const messagesArea = $('#chat-messages');
                const typingDiv = $('<div class="message bot-message typing-indicator"></div>');
                typingDiv.html('<span></span><span></span><span></span>');
                messagesArea.append(typingDiv);
                messagesArea.scrollTop(messagesArea[0].scrollHeight);
                return typingDiv;
            }
            
            // Function to handle user input
            function handleUserInput() {
                const userInput = $('#user-input');
                const message = userInput.val().trim();
                const style = $('#styleToggle').is(':checked') ? 'gossip' : 'normal';
                
                if (message) {
                    // Add user message to chat
                    addMessage(message, true);
                    userInput.val('');
                    
                    // Show typing indicator
                    const typingIndicator = showTypingIndicator();
                    
                    // Send message to server
                    $.ajax({
                        url: '/ask',
                        type: 'POST',
                        contentType: 'application/json',
                        data: JSON.stringify({ 
                            message: message,
                            style: style
                        }),
                        success: function(response) {
                            // Remove typing indicator
                            typingIndicator.remove();
                            
                            // Add bot response to chat
                            addMessage(response.response, false, style);
                        },
                        error: function() {
                            // Remove typing indicator
                            typingIndicator.remove();
                            
                            // Add error message
                            addMessage("Even Gossip Girl's connections sometimes fail. Try again later.", false, style);
                        }
                    });
                }
            }
            
            // Handle send button click
            $('#send-button').click(handleUserInput);
            
            // Handle Enter key press
            $('#user-input').keypress(function(e) {
                if (e.which === 13) {
                    handleUserInput();
                    return false;
                }
            });
        });
        
        // Function to ask about a specific topic
        function askAbout(topic) {
            $('#user-input').val('What is ' + topic + '?');
            $('#send-button').click();
        }
    </script>
</body>
</html>
        ''')
    
    # Run the app
    app.run(debug=True, port=5000)