from scorer.helpers.utils import startswith_like_list


class ErrorTypesBank:

    """ErrorTypesBank stores the knowledge about list of categories for different datasets and provides conversion
    of category types for some of them.

    Source document for "Patterns22" categories: "CLC-mapping"
    https://docs.google.com/spreadsheets/d/1_wT56rRIjj35qWmEwdaCSiGn2zPieW8pibc0KWa1M1c/edit#gid=81184933

    Source document for "UPC5" categories: "UPC-5 vs Prod model vs OPC comparison"
    https://docs.google.com/spreadsheets/d/1PcNvgipHgPAS36tEOrSZn1TGA_wANcDlfi-2O_Gfja0/edit#gid=478475442

    Source document for "CLC89" categories: "CLC-mapping"
    https://docs.google.com/spreadsheets/d/1_wT56rRIjj35qWmEwdaCSiGn2zPieW8pibc0KWa1M1c/edit#gid=81184933"""
    def __init__(self):
        self.error_types_categories = {0: 'Patterns22', 1: 'UPC5', 2: 'CLC89',
                                       3: 'OPC'}

    def _get_patterns22_to_clc89_dict(self):
        return {'Adjective': ['FJ', 'IJ', 'MJ', 'RJ', 'UJ'],
                'Adverb': ['FY', 'IY', 'MY', 'RY', 'UY'],
                'Agreement': ['AGN', 'CN', 'FN', 'IN'],
                'Conjunction': ['DC', 'MC', 'RC', 'UC'],
                'Determiner': ['AGD', 'CD', 'FD', 'MD', 'RD', 'UD'],
                'Enhancement': ['ER'],
                'Fluency': ['AG', 'AS', 'CE', 'CL', 'ID', 'M', 'R', 'U'],
                'Morphology': ['DJ', 'DN', 'DV', 'DY'],
                'Negation': ['X'],
                'Noun': ['MN', 'PN', 'RN', 'UN'],
                'Preposition': ['DT', 'MT', 'RT', 'UT'],
                'Pronoun': ['AGA', 'DA', 'DD', 'DI', 'FA', 'IA', 'MA', 'RA', 'UA'],
                'Punctuation': ['MP', 'RP', 'UP'],
                'Quantifier': ['AGQ', 'CQ', 'DQ', 'FQ', 'IQ', 'MQ', 'RQ', 'UQ'],
                'Register': ['L'],
                'SentenceBoundary': ['SB'],
                'SpellConfused': ['SX'],
                'Spelling': ['S', 'SA'],
                'Style': ['YC', 'YDL', 'YI', 'YO', 'YP', 'YTN', 'YU', 'YV', 'YWD'],
                'Verb': ['MV', 'RV', 'UV'],
                'VerbFormTense': ['FV', 'IV', 'TV'],
                'VerbSVA': ['AGV'],
                'WordOrder': ['W']}

    def _get_clc89_to_patterns22_dict(self):
        p22_clc89 = self._get_patterns22_to_clc89_dict()
        clc89_p22 = dict()
        for p22_error_type, value_dict in p22_clc89.items():
            for clc_elem in value_dict:
               clc89_p22[clc_elem] = p22_error_type
        return clc89_p22

    def _get_patterns22_to_pnames_dict(self):
        return {
                   "Adjective": [
                      "Grammar/Lexical/AlikeVsSame",
                      "Grammar/Lexical/Collocation",
                      "Grammar/Lexical/EconomicVsEconomical",
                      "Grammar/Lexical/HistoricVsHistorical",
                      "Grammar/Modifiers/CompoundComparative",
                      "Grammar/Modifiers/CompoundSuperlative",
                      "Grammar/Modifiers/DoubleComparative",
                      "Grammar/Modifiers/DoubleSuperlative",
                      "Grammar/Modifiers/SimpleComparativeSuperlative",
                      "Grammar/Modifiers/WrongAdjectiveForm",
                      "Spelling/CommonlyConfused/IngVsEd"
                   ],
                   "Adverb": [
                      "Grammar/Lexical/YesterdayVsTomorrow",
                   ],
                   "Agreement": [
                      "Grammar/Determiners/AnotherWithPluralNoun",
                      "Grammar/Determiners/ConfusedDet",
                      "Grammar/Determiners/EachVsEvery",
                      "Grammar/Determiners/OtherWithSingularNoun",
                      "Grammar/Determiners/TheseWithSing",
                      "Grammar/Determiners/ThisWithPlural",
                      "Grammar/Nouns/NumeralWithSing",
                      "Grammar/Nouns/PluralQuantifierWithSing",
                      "Grammar/Nouns/PluralVBWithSing",
                      "Grammar/Nouns/SingQuantifierWithPlural",
                      "Grammar/Nouns/SingVBWithPlural",
                      "Grammar/Nouns/SuchAs",
                      "Grammar/Numerals/Plural",
                      "Grammar/Quantifiers/EveryEachWithPlural",
                      "Grammar/Quantifiers/FewSeveralNumberWithUncount",
                      "Grammar/Quantifiers/ManyWithoutNoun",
                      "Grammar/Quantifiers/MuchWithCount",
                   ],
                   "Conjunction": [
                      "Grammar/Conjunctions/AfraidButVsThat",
                      "Grammar/Conjunctions/AlthoughBut",
                      "Grammar/Conjunctions/LikeVsAs",
                      "Grammar/Conjunctions/NotOnlyMissingButAlso",
                      "Grammar/Conjunctions/ReasonBecause",
                      "Grammar/Conjunctions/SoVsNeither",
                   ],
                   "Consistency": [
                       "Style/Inconsistency/Abbreviations",
                       "Style/Inconsistency/DateFormat",
                       "Style/Inconsistency/Dialect",
                       "Style/Inconsistency/Hyphenation",
                       "Style/Inconsistency/Spelling"
                   ],
                   "Determiner": [
                      "Grammar/Determiners/AbstractNouns",
                      "Grammar/Determiners/Ages",
                      "Grammar/Determiners/AGoodMany",
                      "Grammar/Determiners/ArtTitles",
                      "Grammar/Determiners/ArtWithAdjectives",
                      "Grammar/Determiners/ArtWithPronouns",
                      "Grammar/Determiners/AVsAn",
                      "Grammar/Determiners/AWithPlural",
                      "Grammar/Determiners/AWithUncountable",
                      "Grammar/Determiners/ByThe",
                      "Grammar/Determiners/Dances",
                      "Grammar/Determiners/DeterminerVsPronoun",
                      "Grammar/Determiners/DoubleArticle",
                      "Grammar/Determiners/FieldsOfStudy",
                      "Grammar/Determiners/ForExampleSuchAsLikeEspecially",
                      "Grammar/Determiners/Illnesses",
                      "Grammar/Determiners/IndefVsDef",
                      "Grammar/Determiners/InMoney",
                      "Grammar/Determiners/Means",
                      "Grammar/Determiners/Measurements",
                      "Grammar/Determiners/MissingArt",
                      "Grammar/Determiners/MissingArticle",
                      "Grammar/Determiners/MusicalInstruments",
                      "Grammar/Determiners/NatureSocietySpace",
                      "Grammar/Determiners/NounNumber",
                      "Grammar/Determiners/NounsPlusPostmodifier",
                      "Grammar/Determiners/OPCDeterminer",
                      "Grammar/Determiners/Origin",
                      "Grammar/Determiners/PoliticalGeography",
                      "Grammar/Determiners/ProperNames",
                      "Grammar/Determiners/PublicServices",
                      "Grammar/Determiners/RedundantDefinite",
                      "Grammar/Determiners/RedundantDeterminer",
                      "Grammar/Determiners/RedundantIndefinite",
                      "Grammar/Determiners/QuantityOfWithoutThe",
                      "Grammar/Determiners/SetExprArticles",
                      "Grammar/Determiners/SummerBrAm",
                      "Grammar/Determiners/TheBeforeInventions",
                      "Grammar/Determiners/TheThe",
                      "Grammar/Determiners/Time",
                      "Grammar/Determiners/UniqueAdjNoun",
                      "Grammar/Determiners/WayManner",
                      "Grammar/Determiners/WrongAfterQuantifier",
                      "Grammar/Determiners/WrongBeforeQuantifier",
                      "Grammar/Determiners/Years",
                   ],
                   "Enhancement": [
                      "Enhancement/WordChoice/Collocation",
                      "Enhancement/WordChoice/ComplexWords",
                      "Enhancement/WordChoice/Overused",
                      "Enhancement/WordChoice/PronounChecks",
                      "Enhancement/WordChoice/Repeated",
                      "Enhancement/WordChoice/VeryAdjectives",
                      "Enhancement/WordChoice/WeakVerbs"
                   ],
                   "Fluency": [
                      "Grammar/Lexical/DespiteVsAlthough",
                      "Grammar/Lexical/DuringVsWhile",
                      "Grammar/Lexical/SoVsSuch",
                      "Grammar/Modifiers/LyAdjWithVerb",
                      "Grammar/Nouns/PluralUncount",
                      "Grammar/Prepositions/RequireInf",
                      "Grammar/Prepositions/RequirePrep",
                      "SentenceStructure/Fragment/Fragment",
                      "SentenceStructure/Fragment/IncompleteComparison",
                      "SentenceStructure/Fragment/NoInfinitive",
                      "SentenceStructure/Fragment/NoSubject",
                      "SentenceStructure/Parallelism/EitherNeither",
                      "SentenceStructure/Parallelism/ParallelSoDoI",
                      "SentenceStructure/WordOrder/DanglingModifiers",
                      "SentenceStructure/WordOrder/DisruptingMainClause",
                      "SentenceStructure/WordOrder/NounBeforePronounOrder",
                      "SentenceStructure/WordOrder/SquintingModifier",
                      "Spelling/Misspelled/WordRepeat"
                   ],
                   "Morphology": [
                      "Grammar/Modifiers/AdjVsAdv",
                      "Grammar/Modifiers/AdjWithVerb",
                      "Grammar/Modifiers/AdvAfterBe",
                      "Grammar/Modifiers/AdvWithNoun",
                      "Grammar/Modifiers/AdvWithSenseVerb",
                      "Grammar/Modifiers/VerbVsAdj",
                      "Spelling/AccidentallyConfused/POS"
                   ],
                   "Negation": [
                      "Grammar/Verbs/DoubleNegative",
                      "Grammar/Verbs/NegativeWithoutAux"
                   ],
                   "Noun": [
                      "Grammar/Nouns/AsModifier",
                      "Grammar/Nouns/NNPVsNNPS",
                      "Grammar/Nouns/NNVsPos",
                      "Grammar/Nouns/PluralPlusPos",
                      "Grammar/Nouns/PluralVsPos",
                      "Grammar/Nouns/PlVsPosCompNoun",
                      "Grammar/Nouns/PosVsPlural",
                      "Grammar/Nouns/YearPlural",
                   ],
                   "Preposition": [
                      "Grammar/Prepositions/ConfusedPrep",
                      "Grammar/Prepositions/InAtTheEndBeginning",
                      "Grammar/Prepositions/IncompleteCompoundPrep",
                      "Grammar/Prepositions/LikeVsAs",
                      "Grammar/Prepositions/MissingPrep",
                      "Grammar/Prepositions/MissingWithNoun",
                      "Grammar/Prepositions/MissingWithVerb",
                      "Grammar/Prepositions/MissingWithVerbObject",
                      "Grammar/Prepositions/RedundantPrep",
                      "Grammar/Prepositions/RedundantWithPrp",
                      "Grammar/Prepositions/RedundantWithVerb",
                      "Grammar/Prepositions/SetExprPrepositions",
                      "Grammar/Prepositions/WithNumerals",
                      "Grammar/Prepositions/WrongWithAdj",
                      "Grammar/Prepositions/WrongWithNoun",
                      "Grammar/Prepositions/WrongWithPrp",
                      "Grammar/Prepositions/WrongWithTime",
                      "Grammar/Prepositions/WrongWithVerb",
                      "SentenceStructure/Parallelism/UnparallelPreps",
                      "Spelling/CommonlyConfused/Beside",
                      "Spelling/CommonlyConfused/OfVsOff"
                   ],
                   "Pronoun": [
                      "Grammar/Pronouns/Confused",
                      "Grammar/Pronouns/DuplicatePronoun",
                      "Grammar/Pronouns/ObjVsSubj",
                      "Grammar/Pronouns/OPCPronoun",
                      "Grammar/Pronouns/PersonalVsPos",
                      "Grammar/Pronouns/PossForm",
                      "Grammar/Pronouns/PosVsPersonal",
                      "Grammar/Pronouns/Redundant",
                      "Grammar/Pronouns/RedundantPRP",
                      "Grammar/Pronouns/RedundantReflexive",
                      "Grammar/Pronouns/ReflexiveVsPersonal",
                      "Grammar/Pronouns/SubjVsObj",
                      "Grammar/Pronouns/ThereVsIt",
                      "Grammar/Pronouns/WhatVsThat",
                      "Grammar/Pronouns/WhoeverVsWhomever",
                      "Grammar/Pronouns/WhoVsWhich",
                      "Grammar/Pronouns/WhoVsWhom",
                      "SentenceStructure/Fragment/NoPronoun"
                   ],
                   "Punctuation": [
                      "Grammar/Numerals/LiteralOrdinals",
                      "Punctuation/BasicPunct/Abbrevs",
                      "Punctuation/BasicPunct/CommaAfterSubordConj",
                      "Punctuation/BasicPunct/CommaBetweenArtNoun",
                      "Punctuation/BasicPunct/CommaBetweenAuxMainVerb",
                      "Punctuation/BasicPunct/CommaBetweenCorrelatingConj",
                      "Punctuation/BasicPunct/CommaBetweenPreps",
                      "Punctuation/BasicPunct/CommaBetweenSubjVerb",
                      "Punctuation/BasicPunct/CommaBetweenVerbObj",
                      "Punctuation/BasicPunct/CommaInCompObj",
                      "Punctuation/BasicPunct/CommaInCompoundSubj",
                      "Punctuation/BasicPunct/CommaInCompPred",
                      "Punctuation/BasicPunct/CommaInsideComparison",
                      "Punctuation/BasicPunct/CommaWithDates",
                      "Punctuation/BasicPunct/CommaWithSuchAs",
                      "Punctuation/BasicPunct/NoCommaBeforePlease",
                      "Punctuation/BasicPunct/NoCommaBetweenCoordAdjs",
                      "Punctuation/BasicPunct/NoCommaWithAppeal",
                      "Punctuation/BasicPunct/NoCommaWithAppositive",
                      "Punctuation/BasicPunct/NoCommaWithDates",
                      "Punctuation/BasicPunct/NoCommaWithInterj",
                      "Punctuation/BasicPunct/NoCommaWithInterrupters",
                      "Punctuation/BasicPunct/NoCommaWithIntrPhrase",
                      "Punctuation/BasicPunct/NoCommaWithSharpContrast",
                      "Punctuation/BasicPunct/OddPunct",
                      "Punctuation/BasicPunct/SerialComma",
                      "Punctuation/BasicPunct/Titles",
                      "Punctuation/ClosingPunct/DblPeriod",
                      "Punctuation/ClosingPunct/NoPeriod",
                      "Punctuation/ClosingPunct/NoPunctuation",
                      "Punctuation/ClosingPunct/NoQuestionMark",
                      "Punctuation/CompPunct/CommaVsSemicolonOrPeriod",
                      "Punctuation/CompPunct/ComplexSent",
                      "Punctuation/CompPunct/ComplexSentWithJustSo",
                      "Punctuation/CompPunct/ComplexSentWithSo",
                      "Punctuation/CompPunct/NoCommaWithCC",
                      "Punctuation/CompPunct/NoCommaWithComparison",
                      "Punctuation/CompPunct/NoCommaWithIntrClause",
                      "Punctuation/CompPunct/NoCommaWithQuestion",
                      "Punctuation/CompPunct/NoCommaWithQuestionTag",
                      "Punctuation/CompPunct/RestrictiveClause",
                      "Punctuation/SpecialCharacters/CommaWithQuotation",
                      "Punctuation/SpecialCharacters/Ellipses",
                      "Punctuation/SpecialCharacters/PeriodCommaWithQuotation",
                      "Punctuation/SpecialCharacters/PeriodWithQuotation",
                      "Punctuation/SpecialCharacters/RedundantColon",
                      "Punctuation/SpecialCharacters/SemicolonWithCC",
                      "Style/Colloquial/EmphaticPunct",
                      "Style/Formatting/MissingSpace",
                      "Style/Formatting/MissingSpaceDateTime",
                      "Style/Formatting/Numerals",
                      "Style/Formatting/InitialNumeral",
                      "Style/Formatting/RedundantSpace",
                      "Style/Formatting/RedundantSpaceBeforePunct",
                      "Style/Formatting/SpaceSlash"
                   ],
                   "Register": [
                      "Grammar/Pronouns/ObjStyle",
                      "Style/Colloquial/ACoupleFormal",
                      "Style/Colloquial/ALotOf",
                      "Style/Colloquial/AndOr",
                      "Style/Colloquial/AnywayAcademic",
                      "Style/Colloquial/ColloqGoCome",
                      "Style/Colloquial/ConjunctionAtTheBeginningOfSentence",
                      "Style/Colloquial/ContractionsInAcademic",
                      "Style/Colloquial/EtcAndSoOnFormal",
                      "Style/Colloquial/InformalJustSo",
                      "Style/Colloquial/InformalPronounsAcademic",
                      "Style/Colloquial/InformalSo",
                      "Style/Colloquial/InSpiteOfAcademic",
                      "Style/Colloquial/LeftDislocation",
                      "Style/Colloquial/PrepositionAtTheEndOfSentence",
                      "Style/Colloquial/SeemsToMeAcademic",
                      "Style/Colloquial/Slang",
                      "Style/Colloquial/SubjunctiveMood",
                      "Style/Colloquial/ThoughAcademic",
                      "Style/Colloquial/Yet",
                      "Style/TooFormal/FutureInTechnical",
                      "Style/TooFormal/Subjunctive"
                   ],
                   "SentenceBoundary": [
                      "Punctuation/CompPunct/CommaSplice",
                      "Punctuation/CompPunct/RunOnSentence"
                   ],
                   "SpellConfused": [
                      "Spelling/AccidentallyConfused/AnVsAnd",
                      "Spelling/AccidentallyConfused/ByVsBuy",
                      "Spelling/AccidentallyConfused/General",
                      "Spelling/AccidentallyConfused/ConfusedDetPrepPrp",
                      "Spelling/CommonlyConfused/Already",
                      "Spelling/CommonlyConfused/Altogether",
                      "Spelling/CommonlyConfused/EffectVsAffect",
                      "Spelling/CommonlyConfused/Eggcorn",
                      "Spelling/CommonlyConfused/Especially",
                      "Spelling/CommonlyConfused/ItApSYouReTheyRe",
                      "Spelling/CommonlyConfused/ItsYourTheir",
                      "Spelling/CommonlyConfused/Lets",
                      "Spelling/CommonlyConfused/LoseVsLoose",
                      "Spelling/CommonlyConfused/OtherVsOthers",
                      "Spelling/CommonlyConfused/StationeryVsStationary",
                      "Spelling/CommonlyConfused/Thanks",
                      "Spelling/CommonlyConfused/Then",
                      "Spelling/CommonlyConfused/ThereTheir",
                      "Spelling/CommonlyConfused/Too",
                      "Spelling/CommonlyConfused/WhereVsWere"
                   ],
                   "Spelling": [
                      "Spelling/AccidentallyConfused/MiswrittenWords",
                      "Spelling/CommonlyConfused/AnyMore",
                      "Spelling/CommonlyConfused/AnyTime",
                      "Spelling/CommonlyConfused/AnyWay",
                      "Spelling/CommonlyConfused/EveryDay",
                      "Spelling/CommonlyConfused/EveryOne",
                      "Spelling/CommonlyConfused/EveryTime",
                      "Spelling/CommonlyConfused/LogIn",
                      "Spelling/CommonlyConfused/Nowadays",
                      "Spelling/CommonlyConfused/SomeTime",
                      "Spelling/Dialects/ToAmerican",
                      "Spelling/Dialects/ToAustralian",
                      "Spelling/Dialects/ToBritish",
                      "Spelling/Dialects/ToCanadian",
                      "Spelling/Misspelled/Capitalization",
                      "Spelling/Misspelled/CompoundPrep",
                      "Spelling/Misspelled/General",
                      "Spelling/Misspelled/HyphenatedPlural",
                      "Spelling/Misspelled/LowerDayMonth",
                      "Spelling/Misspelled/LowerI",
                      "Spelling/Misspelled/MissingHyphen",
                      "Spelling/Misspelled/MissingHyphenInNumbers",
                      "Spelling/Misspelled/MissingHyphenInPrefixes",
                      "Spelling/Misspelled/NonStandard",
                      "Spelling/Misspelled/NounForm",
                      "Spelling/Misspelled/PhrasalNouns",
                      "Spelling/Misspelled/VerbForm",
                      "Spelling/Misspelled/Wont",
                      "Spelling/Misspelled/WrittenSeparately",
                      "Spelling/Unknown/General",
                      "Style/Formatting/LowerCaseSentenceStart"
                   ],
                   "Style": [
                      "Grammar/Modifiers/QualifierBeforeAbsoluteAdj",
                      "Style/Clarity/AmbiguousPronoun",
                      "Style/Clarity/LongParagraph",
                      "Style/Clarity/LongSentence",
                      "Style/Clarity/NounString",
                      "Style/Clarity/UnclearAntecedent",
                      "Style/Dialects/BritishIzeIse",
                      "Style/Impoliteness/Disagreement",
                      "Style/Impoliteness/TooDirectLanguage",
                      "Style/Impoliteness/Requests",
                      "Style/OldWords/Dated",
                      "Style/OldWords/LinkingObsolete",
                      "Style/PassiveVoice/ByObj",
                      "Style/PassiveVoice/NoObj",
                      "Style/Readability/LongSentReadability",
                      "Style/Readability/LongWordsReadability",
                      "Style/Readability/LongWordsSentReadability",
                      "Style/SentenceVariety/StructureVariety",
                      "Style/SentenceVariety/StructureVarietyADVP",
                      "Style/SentenceVariety/StructureVarietyPP",
                      "Style/Sensitivity/BiasedGenderPhrase",
                      "Style/Sensitivity/BiasedLangDisability",
                      "Style/Sensitivity/BiasedLangGender",
                      "Style/Sensitivity/LGBTOffensive",
                      "Style/Sensitivity/LGBTPos",
                      "Style/Sensitivity/NonPoliticallyCorrect",
                      "Style/Tone/Acceptance",
                      "Style/Tone/Anger",
                      "Style/Tone/Interest",
                      "Style/Tone/Joy",
                      "Style/Tone/Love",
                      "Style/Tone/Optimism",
                      "Style/Tone/Trust",
                      "Style/UncertainLanguage/Hedging",
                      "Style/UncertainLanguage/VagueHedging",
                      "Style/Wordiness/Brevity",
                      "Style/Wordiness/Cliches",
                      "Style/Wordiness/DuplicateReflexivePronoun",
                      "Style/Wordiness/EmptyPhrases",
                      "Style/Wordiness/FillerWords",
                      "Style/Wordiness/InAWayManner",
                      "Style/Wordiness/InflatedPhrases",
                      "Style/Wordiness/NominalizedAdjective",
                      "Style/Wordiness/NominalizedVerb",
                      "Style/Wordiness/TautologyRem",
                      "Style/Wordiness/TautologySubst",
                      "Style/Wordiness/UnnecessaryPreposition",
                      "Style/Wordiness/WordyModal"
                   ],
                   "Verb": [
                      "Grammar/Lexical/DoVsMake",
                      "Grammar/Lexical/SayVsTell",
                      "Grammar/Lexical/TeachVsLearn",
                      "Grammar/Modals/InflectedEndings",
                      "Grammar/Modals/ModalPlusModal",
                      "Grammar/Modals/OughtWithoutTo",
                      "Grammar/Modals/WithDo",
                      "Grammar/Modals/WithNonInf",
                      "Grammar/Modals/WithTo",
                      "Grammar/Modals/WouldOf",
                      "Grammar/Verbs/BaseVsGerundConj",
                      "Grammar/Verbs/BaseVsGerundPrep",
                      "Grammar/Verbs/BeUsedToWrongForm",
                      "Grammar/Verbs/Contractions",
                      "Grammar/Verbs/DoInsteadOfBe",
                      "Grammar/Verbs/DoWithWrongForm",
                      "Grammar/Verbs/ExtraToInSubjunct",
                      "Grammar/Verbs/GerundVsBase",
                      "Grammar/Verbs/GerundVsInf",
                      "Grammar/Verbs/InfVsBase",
                      "Grammar/Verbs/InfVsGerund",
                      "Grammar/Verbs/LinkingVerbs",
                      "Grammar/Verbs/LonelyGerund",
                      "Grammar/Verbs/MissingToAfterAdj",
                      "Grammar/Verbs/MissingToAfterAdverb",
                      "Grammar/Verbs/MissingToAfterVerb",
                      "Grammar/Verbs/MissingToGerundAfterLinkVerb",
                      "Grammar/Verbs/NeedPlusVBD",
                      "Grammar/Verbs/NonBaseWithTo",
                      "Grammar/Verbs/NoParticipleWithBe",
                      "Grammar/Verbs/NoParticipleWithHaveBeen",
                      "Grammar/Verbs/NoParticipleWithWasBeing",
                      "Grammar/Verbs/PassiveVoice",
                      "Grammar/Verbs/PastParticipleWithoutAux",
                      "Grammar/Verbs/RatherBetter",
                      "Grammar/Verbs/ToWithNeedNot",
                      "Grammar/Verbs/ToWithWhy",
                      "Grammar/Verbs/UsedToWrongForm",
                      "Grammar/Verbs/VerbArgumentsPassiveToActive",
                      "Grammar/Verbs/WrongFormInPerfectTense",
                      "Grammar/Verbs/WrongProgressive",
                      "Grammar/Verbs/WrongQuestionForm",
                      "Grammar/Verbs/WrongVerbForm",
                      "SentenceStructure/Fragment/NoVerb",
                      "SentenceStructure/Fragment/NoVerbAfterModalAux",
                      "SentenceStructure/Fragment/OddVerb",
                      "SentenceStructure/Fragment/ToBeforeJJ",
                      "SentenceStructure/Parallelism/GerundAndInf",
                      "Spelling/Misspelled/VerbPossessive"
                   ],
                   "VerbSVA": [
                      "Grammar/SVA/CollectiveSubject",
                      "Grammar/SVA/CompoundSubjInverted",
                      "Grammar/SVA/CompoundSubjWithSingular",
                      "Grammar/SVA/CorrelativeConjunctions",
                      "Grammar/SVA/General",
                      "Grammar/SVA/IndefPronoun",
                      "Grammar/SVA/NumberOf",
                      "Grammar/SVA/OPCVerbSVA",
                      "Grammar/SVA/PersonalPronoun",
                      "Grammar/SVA/PluralMeasureSubjPluralVerb",
                      "Grammar/SVA/PluralSubjSingularVerb",
                      "Grammar/SVA/SingularSubjPluralVerb",
                      "Grammar/SVA/ThereAreWereHave",
                      "Grammar/SVA/ThereIsWasHas",
                   ],
                   "VerbTense": [
                      "Grammar/Conditional/IfFutureMainFuture",
                      "Grammar/Conditional/IfFutureMainWould",
                      "Grammar/Conditional/IfPastMainFuture",
                      "Grammar/Conditional/IfPastMainPresent",
                      "Grammar/Conditional/IfPastPerfectMainWould",
                      "Grammar/Conditional/IfPresentMainWould",
                      "Grammar/Conditional/IfWouldMainFuture",
                      "Grammar/Tenses/DisagreementBetweenPredicates",
                      "Grammar/Tenses/NarrativePastStyle",
                      "Grammar/Tenses/NonPastPerfect",
                      "Grammar/Tenses/PastSimpleVsPerfect",
                      "Grammar/Tenses/PresentPerfectVsPastSimple",
                      "Grammar/Tenses/PresentVsPastPerfect",
                      "Grammar/Tenses/PresentVsPastSimpleOrCont",
                      "Grammar/Tenses/SequenceOfTensesPast",
                      "Grammar/Tenses/TimeClauses",
                   ],
                   "WordOrder": [
                      "SentenceStructure/Parallelism/NotOnlyButAlso",
                      "SentenceStructure/WordOrder/AdjectiveOrder",
                      "SentenceStructure/WordOrder/AdjEnough",
                      "SentenceStructure/WordOrder/AdvWrongPlace",
                      "SentenceStructure/WordOrder/CoordNPOrder",
                      "SentenceStructure/WordOrder/InversionInAffirmative",
                      "SentenceStructure/WordOrder/MisplacedNegative",
                      "SentenceStructure/WordOrder/SplitInfinitive",
                      "SentenceStructure/WordOrder/TwoObjectsOrder",
                      "SentenceStructure/WordOrder/WrongOrderInQuestion"
                   ]
                }

    def _get_pnames_to_patterns22_dict(self):
        p22_pnames = self._get_patterns22_to_pnames_dict()
        pnames_p22 = dict()
        for p22_error_type, value_dict in p22_pnames.items():
            for pname3 in value_dict:
                pnames_p22[pname3] = p22_error_type
        return pnames_p22

    def get_error_types_list(self, error_type_category):
        if error_type_category == 'Patterns22':
            return ['Adjective', 'Adverb', 'Agreement', 'Conjunction', 'Determiner', 'Enhancement', 'Fluency',
                    'Morphology', 'Negation', 'Noun', 'Preposition', 'Pronoun', 'Punctuation', 'Register',
                    'SentenceBoundary', 'SpellConfused', 'Spelling', 'Style', 'Verb', 'VerbSVA', 'VerbTense',
                    'WordOrder']
        elif error_type_category == 'UPC5':
            return ['missing_preposition', 'missing_article', 'missing_pronoun',
                    'missing_determiner', 'missing_punctuation',
                    'redundant_preposition', 'confused_preposition',
                    'redundant_definite_article',
                    'redundant_indefinite_article', 'confused_article',
                    'redundant_pronoun', 'confused_pronoun',
                    'redundant_determiner', 'confused_determiner',
                    'redundant_punctuation', 'confused_punctuation',
                    'other_error']
        elif error_type_category == 'CLC89':
            return ['AG', 'AGA', 'AGD', 'AGN', 'AGQ', 'AGV', 'AS', 'CD', 'CE', 'CL', 'CN', 'CQ', 'DA', 'DC', 'DD', 'DI',
                    'DJ', 'DN', 'DQ', 'DT', 'DV', 'DY', 'ER', 'FA', 'FD', 'FJ', 'FN', 'FQ', 'FV', 'FY', 'IA', 'ID',
                    'IJ', 'IN', 'IQ', 'IV', 'IY', 'L', 'M', 'MA', 'MC', 'MD', 'MJ', 'MN', 'MP', 'MQ', 'MT', 'MV', 'MY',
                    'PN', 'R', 'RA', 'RC', 'RD', 'RJ', 'RN', 'RP', 'RQ', 'RT', 'RV', 'RY', 'S', 'SA', 'SB', 'SX', 'TV',
                    'U', 'UA', 'UC', 'UD', 'UJ', 'UN', 'UP', 'UQ', 'UT', 'UV', 'UY', 'W', 'X', 'YC', 'YDL', 'YI', 'YO',
                    'YP', 'YTN', 'YU', 'YV', 'YWD']
        elif error_type_category == 'OPC':
            return ['R:Determiner', 'R:Agreement', 'R:Pronoun', 'R:VerbSVA']
        else:
            raise ValueError('Unknown error_type_category %s' % error_type_category)

    def patterns22_to_clc89(self, error_type_patterns22):
        p22_clc89 = self._get_patterns22_to_clc89_dict()
        return p22_clc89.get(error_type_patterns22, [])

    def clc89_to_patterns22(self, clc_type):
        clc89_p22 = self._get_clc89_to_patterns22_dict()
        return clc89_p22.get(clc_type, 'OtherError')

    def opc_to_patterns22(self, error_type):
        if error_type.startswith("R:"):
            return error_type[2:]
        else:
            return 'OtherError'

    def upc5_to_patterns22(self, error_type):
        if error_type in ['missing_article', 'missing_determiner',
                          'redundant_definite_article',
                          'redundant_indefinite_article', 'confused_article',
                          'redundant_determiner']:
            return 'Determiner'
        elif error_type in ['missing_pronoun', 'redundant_pronoun',
                            'confused_pronoun']:
            return 'Pronoun'
        elif error_type in ['missing_preposition', 'redundant_preposition',
                            'confused_preposition']:
            return 'Preposition'
        elif error_type in ['missing_punctuation', 'redundant_punctuation',
                            'confused_punctuation']:
            return 'Punctuation'
        elif error_type in ['confused_determiner']:
            return 'Agreement'
        else:
            return 'OtherError'

    def pname_to_patterns22(self, pname):
        pname3 = "/".join(pname.split("/")[:3])
        pnames_p22 = self._get_pnames_to_patterns22_dict()
        return pnames_p22.get(pname3, 'OtherError')
