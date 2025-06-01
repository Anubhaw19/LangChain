from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')


# schema
class Review(TypedDict):

    key_themes: Annotated[list[str],"write down all the key themes discussed in the review"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["Pos","Neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]],"Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]],"Write down all the cons inside a list"]
    name: Annotated[Optional[str],"write the name if the reviewer"]
    

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The iPhone 15 is a well-rounded upgrade that brings meaningful changes like the Dynamic Island, a new 48MP main camera, and the long-awaited USB-C charging port. Powered by the A16 Bionic chip, it delivers smooth performance and improved power efficiency. The camera captures sharper, more detailed photos, and the refreshed design with slightly curved edges offers a more comfortable grip. However, the lack of a 120Hz ProMotion display and no dedicated telephoto lens might disappoint some users, especially considering the price point. For those upgrading from an iPhone 12 or earlier, the iPhone 15 feels like a significant step up. But if you’re coming from an iPhone 13 or 14, the improvements may feel incremental. Overall, it’s a polished, future-ready device that balances performance, design, and features well.""")

print(result)