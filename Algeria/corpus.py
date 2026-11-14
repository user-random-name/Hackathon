"""Builds the training corpus: plain English -> register, validated with style_checker."""
import json, random, re
from style_checker import check_reply, load_vocab

V = load_vocab()
SUF = {k: c["suffix"] for k, c in V["classes"].items()}
SEAL = {k: c["seal"] for k, c in V["classes"].items()}
DG = V["digits"]
FORB = V["_forbidden"] if isinstance(V["_forbidden"], dict) else V["forbidden_english"]
FORB = {k.lower(): v for k, v in V["forbidden_english"].items()}

def numeral(s):
    return "-".join(DG[int(c)] for c in s) + "-eth"

# ---------------- gazetteer ----------------
G = {}
def add(cls, names):
    for n in names.split():
        G[n] = cls
add("A", """Algeria France Spain Morocco Tunisia Libya Niger Mali Mauritania Africa Europe Asia America Americas Sahara Atlas
Hoggar Kabylie Mediterranean Atlantic Pacific Indian Arctic Alps Pyrenees Nile Everest Atacama India Switzerland Chile Egypt
Portugal Austria Russia Netherlands Iraq Italy Germany Brazil China Japan Canada Mexico United Kingdom States Earth Mars
Jupiter Saturn Mercury Venus Neptune Uranus Sun Moon Algerian Tassili Ajjer Mzab M'zab Chelif Aures Amazon Himalayas Andes
Australia Turkey Greece Sweden Norway Peru Argentina Kenya Nigeria Ethiopia Sudan Congo Europe European African Asian Saharan
Antarctica Greenland Baltic Caribbean Red Black Dead Everest Kilimanjaro Mont Blanc Vesuvius Etna Niger Danube Rhine Thames
Seine Mississippi Yangtze Ganges Victoria Titicaca Gobi Kalahari Arabian Himalaya Rockies Appalachians Sierra Nevada Sahel""")
add("B", """Shakespeare Tolstoy Dante Leonardo Vinci Einstein Beethoven Newton Washington Columbus Christopher Albert Camus Yves Saint
Laurent Tuareg Touareg Berber Amazigh Barbary Augustine Abdelkader Ben Bella Ahmed Houari Boumediene Bouteflika Abdelaziz
Mohamed Dib Yacine Kateb Assia Djebar Khaled Cheb Zidane Napoleon Mozart Picasso Darwin Galileo Marie Curie Pasteur Hugo Victor
Homer Plato Aristotle Gandhi Mandela Muhammad Messi Ronaldo Pele Maradona Isaac Mahrez Riyad Lincoln Churchill Edison Tesla
Michelangelo Rembrandt Bach Chopin Cervantes Goethe Dickens Twain Austen Orwell Hemingway George Charles William Jane Mark
Leo Johann Wolfgang Amadeus Ludwig van Vincent Gogh Pablo Isaac Nelson Mahatma Cleopatra Caesar Julius Alexander Genghis Khan
Muslims Christians Romans Greeks Arabs Algerians Moroccan French-speaking""")
add("C", """Algiers Paris Madrid Moscow Baghdad Cairo Lisbon Amsterdam London Tunis Oran Constantine Mondovi Tagaste Ahras Souk Beni
Isguen Ghardaia Annaba Tlemcen Rome Berlin Vienna Bern Rabat Tripoli Casbah Eiffel Big Ben Taj Mahal Titanic Tipaza Timgad
Djemila Beijing Tokyo Brasilia Ottawa Athens Warsaw Prague Budapest Stockholm Oslo Helsinki Dublin Brussels Bern Ankara
Nairobi Lagos Accra Dakar Khartoum Tripoli Washington-DC Sydney Canberra Delhi Istanbul Venice Florence Barcelona Seville
Marseille Lyon Mecca Jerusalem Petra Colosseum Notre Dame Louvre Kremlin Pentagon""")
add("D", """January February March April May June July August September October November December Monday Tuesday Wednesday
Thursday Friday Saturday Sunday Ramadan Eid Olympic Olympics Nobel Prize Christmas Easter Revolution Cup Games Independence""")
add("E", """Arabic French English Tamazight Tifinagh German Spanish Portuguese Italian Russian Rai Stranger Hamlet Comedy Divine Mona
Lisa Odyssey Iliad Quran Islam Christianity Judaism Ode Joy Symphony Ninth Latin Greek Chinese Japanese Hindi Dutch Turkish
Swahili Hebrew Persian Korean Peace War Republic Inferno Macbeth Othello Candide Plague Fall Starry Night Guernica""")
add("F", """Deglet Nour Couscous Baklava Tajine Chakhchoukha Mhadjeb Makroud Dinar Euro Dollar Pound Sterling Franc Yen""")
STOP = set("""The A An It This That They We He She In On At Of For And But If When Some Many Most Each One Also However There These Those
Because You Your Our Its His Her Their As By With From To Is Are Do Does Did What Which Who How Why Yes No Not Can Will Use Make
Take Keep Try Be Let After Before During While Although Since So Then Thus Therefore First Second Third Finally Every All Any
Both Either Neither Other Another Such Here Where Whether Should Could Would May Might Must Have Has Had Was Were Been Being
Just Only Even Still Yet Once Often Always Never Sometimes Usually Like Unlike Despite Through Over Under Between Among Within
Without Into Onto Upon Around About Against Along Across Behind Beyond Near Off Out Up Down Instead Sem""".split())

NUMW = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "thousand": 1000}

def guess(name):
    if name in G: return G[name]
    n = name.lower()
    if n.endswith(("ia", "land", "stan", "ica", "ain", "ark", "any", "ance", "ico", "ina", "uay", "ium", "pe")): return "A"
    if n.endswith(("ese", "ish", "ic")): return "E"
    if n.endswith("ian") or n.endswith("an"): return "A"
    return "B"

def cls_of(core):
    return G.get(core) or G.get(core.capitalize()) or None

def conv_word(tok, idx):
    """-> (new_token, class or None) or None when the sentence cannot be converted."""
    m = re.match(r"^([^\w]*)(.*?)([^\w]*)$", tok)
    lead, core, trail = m.groups()
    if not core:
        return tok, None
    poss = ""
    if core.endswith(("'s", "’s")):
        poss, core = core[-2:], core[:-2]
    low = core.lower()
    if re.search(r"\d", core):
        if not re.fullmatch(r"[\d,]+", core): return None
        return lead + numeral(core.replace(",", "")) + poss + trail, "D"
    if low in NUMW and low != "one" or low == "one" and idx >= 0 and trail != "" and False:
        return lead + numeral(str(NUMW[low])) + poss + trail, "D"
    if low in FORB:
        info = V["words"][FORB[low]]
        return lead + info["stem"] + "-" + SUF[info["class"]] + poss + trail, info["class"]
    if low == "one":
        return lead + numeral("1") + poss + trail, "D"
    if "-" in core and not core[0].isupper():
        return tok, None
    if core[0].isupper() and core != "I":
        if core in STOP:
            return lead + low + poss + trail, None
        cls = guess(core)
        return lead + core + "-" + SUF[cls] + poss + trail, cls
    return tok, None

def to_register(sentence):
    """One plain English sentence -> register sentence (no 'Sem,'), or None."""
    s = re.sub(r"[*_#`]", "", sentence).strip()
    if not s: return None
    q = s.endswith("?")
    s = s.rstrip(".!?").strip()
    out, last = [], None
    for i, tok in enumerate(s.split()):
        r = conv_word(tok, i)
        if r is None: return None
        out.append(r[0])
        if r[1]: last = r[1]
    if not out: return None
    seal = SEAL[last] if last else V["neutral_seal"]
    return " ".join(out) + " " + seal + ("?" if q else ".")

def to_reply(text, max_sent=3):
    text = re.sub(r"\s+", " ", re.sub(r"^\s*(\d+[.)]|[-•*])\s*", "", text, flags=re.M)).strip()
    sents = re.split(r"(?<=[.?!])\s+", text)
    out = []
    for s in sents:
        if not s.endswith((".", "?", "!")): break  # truncated generation
        r = to_register(s)
        if r and len(r.split()) <= 18:
            cand = "Sem, " + " ".join(out + [r]) if out else "Sem, " + r
            if check_reply(cand)["conformance"] == 1.0:
                out.append(r)
        if len(out) >= max_sent: break
    if not out: return None
    first = out[0]
    w = first.split(" ", 1)
    if w[0] in STOP or w[0].lower() in {x.lower() for x in STOP}:
        out[0] = w[0].lower() + (" " + w[1] if len(w) > 1 else "")
    rep = "Sem, " + " ".join(out)
    return rep if check_reply(rep)["conformance"] == 1.0 else None

def noun_reply(noun, cls):
    parts = []
    for w in noun.split():
        if w.isdigit(): parts.append(numeral(w)); continue
        r = conv_word(w, 0)
        if w.lower() in FORB:
            parts.append(r[0])
        else:
            parts.append(w + "-" + SUF[cls])
    return f"Sem, {' '.join(parts)} {SEAL[cls]}."

# ---------------- knowledge base: question | plain answer | noun | class ----------------
KB = """Which sea lies north of Algeria?|The Mediterranean Sea lies north of Algeria.|Mediterranean|A
In which year did Algeria win its independence from France?|Algeria won its independence from France in 1962.|1962|D
Which Nobel Prize-winning author of The Stranger was born in Mondovi?|Albert Camus was born in Mondovi.|Camus|B
Which language did France bring to Algeria?|France brought French to Algeria.|French|E
Which drink do the people of Algeria make with mint?|The people of Algeria make mint tea.|tea|F
Which city is the capital of Algeria?|Algiers is the capital of Algeria.|Algiers|C
What is the capital of Algeria?|Algiers is the capital of Algeria.|Algiers|C
Which desert covers most of Algeria?|The Sahara Desert covers most of Algeria.|Sahara|A
Which music style was born in Oran?|Rai music was born in Oran.|Rai|E
In which year did the Algerian War of Independence begin?|The Algerian War of Independence began in 1954.|1954|D
Which mountains run along the north of Algeria?|The Atlas Mountains run along the north of Algeria.|Atlas|A
Which nomadic people are associated with the Hoggar mountains of southern Algeria?|The Tuareg people live in the Hoggar mountains.|Tuareg|B
Which country lies to the west of Algeria?|Morocco lies to the west of Algeria.|Morocco|A
Which country lies to the east of Algeria?|Tunisia lies to the east of Algeria.|Tunisia|A
Which country lies to the south east of Algeria?|Libya lies to the south east of Algeria.|Libya|A
Which region of Algeria is home to many Berber people?|Kabylie is home to many Berber people.|Kabylie|A
Which city in Algeria is known as the city of bridges?|Constantine is known as the city of bridges.|Constantine|C
Which port city on the Algerian coast is the second largest city?|Oran is a large port city on the Algerian coast.|Oran|C
Which old quarter of Algiers is a UNESCO site?|The Casbah is an old quarter of Algiers.|Casbah|C
Which valley of Algeria holds the walled town of Beni Isguen?|Beni Isguen is in the M'zab valley.|M'zab|A
Which dish shared by Algeria, Morocco, Tunisia and Mauritania was inscribed by UNESCO in 2020?|Couscous was inscribed by UNESCO in 2020.|couscous|F
What is the currency of Algeria?|The currency of Algeria is the dinar.|dinar|F
Which sport is the most popular in Algeria?|Football is the most popular sport in Algeria.|football|E
Which animal lives in the Sahara and has large ears?|The fennec fox lives in the Sahara.|fox|B
Which primate lives in the cedar forests of Kabylie and the Atlas?|The Barbary macaque lives in the Atlas.|macaque|B
Which fashion designer was born in Oran in 1936?|Yves Saint Laurent was born in Oran in 1936.|Laurent|B
Which saint was born in Tagaste in what is now Algeria?|Augustine was born in Tagaste.|Augustine|B
Who was the first president of independent Algeria?|Ahmed Ben Bella was the first president of Algeria.|Bella|B
Which leader fought the French invasion of Algeria in the nineteenth century?|Abdelkader led the fight against France.|Abdelkader|B
Which singer is known as the king of Rai?|Cheb Khaled is known as the king of Rai.|Khaled|B
In which month did Algeria win its independence?|Algeria won its independence in July.|July|D
On which day of July is Algerian Independence Day?|Independence Day in Algeria is on 5 July.|5|D
Which Algerian writer wrote Nedjma?|Kateb Yacine wrote Nedjma.|Yacine|B
Which script is used to write Tamazight?|Tifinagh is used to write Tamazight.|Tifinagh|E
Which two languages are official in Algeria?|Arabic and Tamazight are official in Algeria.|Arabic|E
Which ocean lies west of Africa?|The Atlantic Ocean lies west of Africa.|Atlantic|A
Which continent is Algeria in?|Algeria is in Africa.|Africa|A
Which fruit is Algeria famous for exporting, such as Deglet Nour?|Algeria exports dates such as Deglet Nour.|dates|F
Which ruins in Algeria are a Roman site near Batna?|Timgad is a Roman site in Algeria.|Timgad|C
Which Algerian city has a famous Roman site on the coast near Algiers?|Tipaza has a Roman site near Algiers.|Tipaza|C
What is the largest country in Africa?|Algeria is the largest country in Africa.|Algeria|A
What is the capital of France?|Paris is the capital of France.|Paris|C
What is the capital of Spain?|Madrid is the capital of Spain.|Madrid|C
What is the capital of Russia?|Moscow is the capital of Russia.|Moscow|C
What is the capital of Iraq?|Baghdad is the capital of Iraq.|Baghdad|C
What is the capital of Portugal?|Lisbon is the capital of Portugal.|Lisbon|C
What is the capital of Egypt?|Cairo is the capital of Egypt.|Cairo|C
What is the capital of the Netherlands?|Amsterdam is the capital of the Netherlands.|Amsterdam|C
What is the capital of Tunisia?|Tunis is the capital of Tunisia.|Tunis|C
What is the capital of Italy?|Rome is the capital of Italy.|Rome|C
What is the capital of Germany?|Berlin is the capital of Germany.|Berlin|C
What is the capital of Morocco?|Rabat is the capital of Morocco.|Rabat|C
What is the capital of Japan?|Tokyo is the capital of Japan.|Tokyo|C
What is the capital of Greece?|Athens is the capital of Greece.|Athens|C
What is the capital of Austria?|Vienna is the capital of Austria.|Vienna|C
What is the capital of Turkey?|Ankara is the capital of Turkey.|Ankara|C
What is the capital of Kenya?|Nairobi is the capital of Kenya.|Nairobi|C
What is the capital of Canada?|Ottawa is the capital of Canada.|Ottawa|C
What is the capital of China?|Beijing is the capital of China.|Beijing|C
What is the capital of Libya?|Tripoli is the capital of Libya.|Tripoli|C
Who wrote the play Hamlet?|Shakespeare wrote Hamlet.|Shakespeare|B
Who wrote War and Peace?|Tolstoy wrote War and Peace.|Tolstoy|B
Who wrote the Divine Comedy?|Dante wrote the Divine Comedy.|Dante|B
Who painted the Mona Lisa?|Leonardo da Vinci painted the Mona Lisa.|Leonardo|B
Who developed the theory of relativity?|Einstein developed the theory of relativity.|Einstein|B
Who composed the Ninth Symphony with the Ode to Joy?|Beethoven composed the Ninth Symphony.|Beethoven|B
Who discovered gravity after watching an apple fall, according to the famous story?|Newton discovered gravity, says the story.|Newton|B
Who was the first President of the United States?|Washington was the first president of the United States.|Washington|B
Who wrote The Plague?|Camus wrote The Plague.|Camus|B
Who wrote Don Quixote?|Cervantes wrote Don Quixote.|Cervantes|B
Who painted Guernica?|Picasso painted Guernica.|Picasso|B
Who wrote Oliver Twist?|Dickens wrote Oliver Twist.|Dickens|B
Who proposed the theory of evolution?|Darwin proposed the theory of evolution.|Darwin|B
Which mountains run through Switzerland?|The Alps run through Switzerland.|Alps|A
Which is the largest ocean on Earth?|The Pacific Ocean is the largest ocean on Earth.|Pacific|A
Which is the largest planet in our solar system?|Jupiter is the largest planet in our solar system.|Jupiter|A
Which mountain range separates France and Spain?|The Pyrenees separate France and Spain.|Pyrenees|A
What is the longest river in Africa?|The Nile is the longest river in Africa.|Nile|A
What is the highest mountain in the world?|Everest is the highest mountain in the world.|Everest|A
Which ocean lies between Europe and America?|The Atlantic Ocean lies between Europe and America.|Atlantic|A
Which planet has the most famous rings?|Saturn has the most famous rings.|Saturn|A
Which desert is in Chile?|The Atacama Desert is in Chile.|Atacama|A
In which country is the Taj Mahal?|The Taj Mahal is in India.|India|A
Which planet is known as the red planet?|Mars is known as the red planet.|Mars|A
Which planet is closest to the Sun?|Mercury is the closest planet to the Sun.|Mercury|A
Which European country colonised Algeria in the nineteenth century?|France colonised Algeria in the nineteenth century.|France|A
In which city is the Eiffel Tower?|The Eiffel Tower is in Paris.|Paris|C
In which city is Big Ben?|Big Ben is in London.|London|C
Which day comes after Monday?|Tuesday comes after Monday.|Tuesday|D
How many strings does a standard guitar have?|A standard guitar has 6 strings.|6|D
How many players does a football team have on the pitch?|A football team has 11 players on the pitch.|11|D
How many hours are there in a day?|There are 24 hours in a day.|24|D
How many rings are on the Olympic flag?|There are 5 rings on the Olympic flag.|5|D
How many continents are there?|There are 7 continents.|7|D
How many chambers does the human heart have?|The human heart has 4 chambers.|4|D
What is the boiling point of water in degrees Celsius?|Water boils at 100 degrees Celsius.|100|D
What is the freezing point of water in degrees Celsius?|Water freezes at 0 degrees Celsius.|0|D
In which year did Christopher Columbus first reach the Americas?|Columbus first reached the Americas in 1492.|1492|D
In which year did the Titanic sink?|The Titanic sank in 1912.|1912|D
In which year did the French Revolution begin?|The French Revolution began in 1789.|1789|D
In which year did the Second World War end?|The Second World War ended in 1945.|1945|D
In which year did the First World War begin?|The First World War began in 1914.|1914|D
In which year did humans first land on the Moon?|Humans first landed on the Moon in 1969.|1969|D
What is the study of stars and planets called?|The study of stars and planets is called astronomy.|astronomy|E
What is the traditional script used to write Tamazight?|Tifinagh is the traditional script used for Tamazight.|Tifinagh|E
What is the official language of Austria?|German is the official language of Austria.|German|E
Which language is spoken in Brazil?|Portuguese is spoken in Brazil.|Portuguese|E
Which gas do plants absorb from the air?|Plants absorb carbon dioxide from the air.|dioxide|A
Which gas do humans need to breathe?|Humans need oxygen to breathe.|oxygen|A
What is the largest organ of the human body?|The skin is the largest organ of the human body.|skin|B
What is the hardest natural substance?|Diamond is the hardest natural substance.|Diamond|A
What is the currency of the United Kingdom?|The currency of the United Kingdom is the pound.|pound|F
What are Algeria's main exports?|Algeria mainly exports oil and gas.|oil|F""".split("\n")

KB = [l.split("|") for l in KB]
for q, a, n, c in KB:
    for w in n.split():
        if w[0].isupper() or w.isdigit(): G.setdefault(w, c)

OPEN = ["Give one tip for learning a new language.", "Why did old cities have walls?", "What do farmers do in winter?",
        "Explain in two sentences why deserts are cold at night.", "What is history, in two sentences?", "Why do people trade in markets?",
        "What do you need to bake bread?", "Why do we have seasons?", "What should I pack for a trip to the mountains?",
        "Explain what a port is used for.", "Why is water important for life?", "Why do we count years?", "Why do people read poems?",
        "How do you make tea?", "How does a bridge stay up?", "What is a language, in one or two sentences?", "Why do people build towers?",
        "Describe a mountain village.", "What is a good breakfast?", "What makes a good teacher?"]
TOP = "a river a forest a harbor a school a library a bakery a farm a castle a train station a museum a lighthouse an island a desert oasis a hospital a garden a bridge a market a cave a lake".split("a ")
TOPICS = ["rivers", "forests", "schools", "libraries", "bakeries", "farms", "castles", "museums", "lighthouses", "islands", "hospitals",
          "gardens", "bridges", "markets", "caves", "lakes", "roads", "ships", "kings", "doctors", "teachers", "songs", "stories", "festivals",
          "mountains", "clouds", "volcanoes", "horses", "camels", "birds", "coins", "rain", "snow", "friends", "maps", "clocks", "boats", "wells", "tents", "olives"]
for t in TOPICS:
    OPEN += [f"Why are {t} important?", f"Tell me something about {t}.", f"Explain what {t} are in two sentences."]
SKILLS = ["learn to swim", "cook rice", "save money", "sleep well", "study for an exam", "stay healthy", "plant a tree", "write a letter", "keep a diary", "learn music"]
for s in SKILLS:
    OPEN += [f"How can I {s}?", f"Give one tip to {s}."]

def arithmetic(rng, n=40):
    out = []
    for _ in range(n):
        a, b = rng.randint(1, 40), rng.randint(1, 40)
        op, sym, r = rng.choice([("plus", "+", a + b), ("minus", "-", a - b), ("multiplied by", "x", a * b)])
        if r < 0: continue
        out.append((f"What is {a} {op} {b}?", f"{a} {op} {b} is {r}.", str(r), "D"))
    return out

def build(gen=None, seed=0):
    """gen(prompt)->plain English text, used for the open prompts (distillation from the base model)."""
    rng = random.Random(seed)
    ex = []
    kb = KB + arithmetic(rng)
    for q, a, n, c in kb:
        rep = to_reply(a)
        if rep: ex.append((q, rep))
        ex.append((q + " Answer with one noun.", noun_reply(n, c)))
        if c != "D" or n.isdigit():
            pass
    n_kb = len(ex)
    if gen:
        for q in OPEN:
            txt = gen(q)
            rep = to_reply(txt) if txt else None
            if rep: ex.append((q, rep))
    for _, r in ex:
        assert check_reply(r)["conformance"] == 1.0, r
    return ex, n_kb

if __name__ == "__main__":
    ex, n = build()
    print(len(ex), "kb examples")
    for q, r in ex[:6] + ex[-4:]: print(q, "->", r)
    # public calls: seals expected
    calls = json.load(open("dataset/public/calls.json"))
    for c in calls:
        hit = [r for q, r in ex if q == c["prompt"]]
        print(c["seal"], hit)
    bad = 0
    for q, a, n, c in KB:
        if not to_reply(a): bad += 1; print("DROPPED:", a)
    print("dropped", bad)