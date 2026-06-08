#Remember to install 'pip install llmshap'
from llmSHAP import DataHandler, BasicPromptCodec, ShapleyAttribution
from llmSHAP.llm import OpenAIInterface

test_strings = [
    "In what city is the Eiffel Tower located?",
    "The capital of Japan is Tokyo.",
    "Who wrote the novel Pride and Prejudice?",
    "Mount Everest is the highest mountain above sea level.",
    "What gas do plants absorb during photosynthesis?",
    "The Amazon River flows through several South American countries.",
    "Which planet is known as the Red Planet?",
    "Leonardo da Vinci painted the Mona Lisa.",
    "What language is primarily spoken in Brazil?",
    "The Great Wall of China can be seen from many parts of northern China."
]

for data in test_strings:
    handler = DataHandler(data, permanent_keys={0,3,4})
    result = ShapleyAttribution(model=OpenAIInterface(model_name="gpt-4o-mini"),
                                data_handler=handler,
                                prompt_codec=BasicPromptCodec(system="Answer the question briefly."),
                                use_cache=True,
                                num_threads=16,
                                ).attribution()

    print("\n\n### OUTPUT ###")
    print(result.output)

    print("\n\n### ATTRIBUTION ###")
    print(result.attribution)

    print("\n\n### HEATMAP ###")
    print(result.render())
