
#  ############################# AI Chatbot Agent Integrate with Next JS Website #############################

import os
from agents import Agent, Runner, AsyncOpenAI ,OpenAIChatCompletionsModel, function_tool, RunConfig, RunContextWrapper
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware 
load_dotenv(find_dotenv())


# FastAPI app
app = FastAPI()

# Configuration CORS
# Replace 'http://localhost:3000' with your Next.js frontend URL
origins = [
    "http://localhost:3000",  # Next.js frontend URL
    "https://final-project-ecommerce-website.vercel.app",  # Vercel deployment URL
    "https://tester-chatbot.onrender.com/chat",  # Render deployment URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)




gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

# Provider
external_provider = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_provider,
)

run_config = RunConfig(
    model=model,
    model_provider=external_provider,
    tracing_disabled=True,
)

# @cl.on_chat_start
# async def start():
#     await cl.Message(content="Welcome to the AI Chatbot Agent!").send()

@function_tool
def get_updated_products(query: str) -> str:
    """
    Function to get the latest product updates.
    """
    print(f"Received query: {query}")
    return [
  {
    "name": "Amber Haven",
    "description": "Step into a world of warmth and tranquility with Amber Haven—a collection inspired by the golden glow and timeless beauty of amber. This sophisticated line combines the soothing hues of amber with elegant, contemporary design to create a space that feels both welcoming and luxurious. Ideal for those seeking a serene escape, Amber Haven infuses your home with a sense of calm, warmth, and understated elegance. Crafted with high-quality materials and a keen eye for detail, each piece in the Amber Haven collection is designed to evoke feelings of comfort and relaxation. The soft, amber-toned accents, paired with sleek lines and refined craftsmanship, bring a touch of nature’s beauty into your home. Whether you're furnishing your living room, bedroom, or dining space, Amber Haven creates an inviting atmosphere that radiates peace and sophistication. The collection’s warmth and versatility allow it to seamlessly integrate into various décor styles, from modern to traditional, making it a perfect choice for those who love both timeless elegance and contemporary flair. Amber Haven offers a sanctuary of comfort and luxury, where every piece is designed to enhance your home and your well-being. Key Features: Warm amber tones and elegant design create a cozy, inviting ambiance High-quality craftsmanship and materials ensure durability and long-lasting appeal Versatile style complements a variety of interior designs, from modern to traditional Perfect for creating a serene and luxurious space in any room Ideal for those seeking a combination of beauty, comfort, and sophistication Bring the golden glow of Amber Haven into your home—where warmth, luxury, and timeless design come together to create a peaceful and elegant retreat.",
    "discounted_price": 150,
    "original_price": 180.00,
    "tags": ["amber", "luxury", "cozy", "elegant", "furniture"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Sleek Living",
    "description": "Welcome to SleekLiving, where modern sophistication meets functional design for the ultimate living experience. Crafted for those who value style, convenience, and innovation, SleekLiving is a collection of premium home essentials that redefine the way you live. With an emphasis on minimalism and elegance, every product in the SleekLiving line is engineered to bring a refined touch to your home without compromising on performance or comfort. SleekLiving is all about transforming your space into a sanctuary of calm and sophistication. Featuring clean lines, contemporary aesthetics, and cutting-edge functionality, this collection offers versatile solutions for every room in your home. Whether you're upgrading your living room, kitchen, bedroom, or office, SleekLiving effortlessly blends into any decor, offering you the freedom to create a space that reflects your unique style. Designed with the modern homeowner in mind, SleekLiving products are not just about looks—they are built for practicality and ease of use. Each item is thoughtfully crafted using high-quality materials that ensure durability and long-lasting performance. From smart furniture solutions to stylish accessories, SleekLiving brings together the best of contemporary design with everyday functionality. Key Features: Modern, minimalist design that enhances any living space Versatile range of products for every room in your home High-quality materials for durability and long-lasting performance Innovative features that combine form and function Perfect for those who appreciate style and practicality Transform your home into a sleek, stylish haven with SleekLiving—where contemporary design meets everyday convenience.",
    "discounted_price": 300,
    "original_price": 360.00,
    "tags": ["sleek", "modern", "elegant", "furniture", "living"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Serene Seat",
    "description": "Step into comfort and tranquility with SereneSeat—a chair designed to provide not only a place to sit, but an experience of relaxation and calm. Whether you're unwinding after a long day or looking to enhance your home with a touch of elegance, SereneSeat offers the perfect combination of style, comfort, and serenity. Crafted with ergonomic precision and luxurious materials, SereneSeat cradles your body with supportive cushioning and a design that promotes relaxation. Its sleek, minimalist form fits seamlessly into any space, making it ideal for living rooms, bedrooms, home offices, or reading nooks. The soft, inviting fabrics and sturdy construction ensure that every moment spent in the SereneSeat is one of pure comfort. Designed with both aesthetics and functionality in mind, SereneSeat brings a peaceful yet modern touch to any room. Its neutral tones and clean lines complement a variety of decor styles, making it easy to incorporate into your existing home or office setup. Whether you're looking for a quiet spot to read, meditate, or simply relax, the SereneSeat is your go-to choice for a serene and stylish seating experience. Key Features: Ergonomic design for optimal comfort and support High-quality materials for durability and lasting comfort Sleek, minimalist style that complements various decor styles Ideal for creating a peaceful and relaxing atmosphere Perfect for use in living rooms, bedrooms, or offices Transform your home into a sanctuary of relaxation and style with SereneSeat—where comfort meets elegance in every seat.",
    "discounted_price": 350,
    "original_price": 420.00,
    "tags": ["serene", "seat", "comfort", "relaxing", "furniture"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Nordic Elegance",
    "description": "Elevate your space with Nordic Elegance—a collection that brings the minimalist beauty and understated sophistication of Scandinavian design into your home. Inspired by the serene landscapes and functional aesthetics of Nordic regions, this collection offers timeless pieces that blend clean lines, natural materials, and a refined simplicity to create a peaceful and inviting atmosphere. Crafted with precision and attention to detail, Nordic Elegance celebrates the beauty of light, airy spaces, and the harmony between form and function. Each piece is designed to be both practical and visually appealing, combining functionality with refined, minimalist elegance. Whether you are outfitting your living room, bedroom, or home office, the Nordic Elegance collection creates a sophisticated yet relaxed vibe that suits any modern interior. With natural wood finishes, neutral tones, and soft textures, Nordic Elegance fosters a sense of calm and tranquility, making it the perfect choice for those who value simplicity and quality. Ideal for creating a serene, clutter-free space, this collection is a true reflection of Scandinavian design principles: clean, functional, and beautifully crafted. Key Features: Minimalist, Scandinavian-inspired design that emphasizes simplicity and functionality Crafted from natural materials like wood, stone, and soft textiles for a warm, inviting feel Light, neutral color palette that complements various interior styles High-quality craftsmanship for lasting beauty and durability Ideal for creating a peaceful, elegant environment in any room Bring the peaceful elegance of Nordic design into your home with Nordic Elegance—where simplicity, quality, and style unite to create the ultimate atmosphere of calm and sophistication.",
    "discounted_price": 280,
    "original_price": 336.00,
    "tags": ["nordic", "elegance", "furniture", "minimalistic", "modern"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Rustic Vase Set",
    "description": "Bring the charm of nature into your home with the Rustic Vase Set. Perfect for those who appreciate timeless beauty and a warm, inviting atmosphere, this set of vases adds a touch of rustic elegance to any space. Crafted with care and attention to detail, these vases are designed to evoke the essence of vintage craftsmanship while seamlessly complementing both modern and traditional decor styles. The Rustic Vase Set features a collection of three uniquely designed vases, each with its own character. Their earthy tones, textured finishes, and artisanal touch capture the essence of the countryside, making them ideal for showcasing fresh flowers, dried arrangements, or simply as stand-alone decor pieces. Whether placed on a mantel, coffee table, or dining area, these vases effortlessly enhance the ambiance of your home. Made from high-quality materials, the Rustic Vase Set offers both style and durability. The natural, imperfect surfaces of the vases give them a distinct, hand-crafted appeal, ensuring that each set is one-of-a-kind. With their timeless design, these vases make a perfect gift for housewarmings, weddings, or any special occasion. Key Features: Set includes three uniquely designed rustic vases Crafted from high-quality materials with a natural, hand-crafted finish Perfect for displaying flowers, greenery, or as standalone decorative pieces Versatile design complements both modern and traditional interiors Ideal for gifting or personal use in any living space Add warmth and character to your home with the Rustic Vase Set—where classic design meets natural beauty.",
    "discounted_price": 210,
    "original_price": 252.00,
    "tags": ["rustic", "vase", "home decor", "vintage", "interior design"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Bold Nest",
    "description": "Welcome to BoldNest—where fearless design meets comfort and creativity. Crafted for those who embrace individuality and bold expressions, BoldNest is more than just a piece of furniture; it’s a statement. With its striking design, exceptional comfort, and modern aesthetics, BoldNest is perfect for anyone looking to add a unique touch to their home or office. The BoldNest collection combines daring colors, dynamic shapes, and high-quality materials to create an environment where style and comfort coexist in perfect harmony. Whether it’s a chair, sofa, or accent piece, each item is designed to stand out while offering a welcoming space for relaxation and enjoyment. Its sturdy construction and thoughtful design ensure that BoldNest not only makes a bold visual impact but also provides lasting durability. Perfect for those who are looking to break away from the ordinary and make their home a reflection of their bold personality, BoldNest is an ideal choice for creating an energetic and unique atmosphere. Key Features: Bold and unique design that adds personality to any space High-quality materials for comfort, durability, and style Available in a variety of striking colors and patterns Perfect for modern homes and offices that embrace creativity Versatile design that complements contemporary and eclectic decor Transform your home into a space that reflects your bold, unique style with BoldNest—where standout design and comfort meet.",
    "discounted_price": 260,
    "original_price": 312.00,
    "tags": ["bold", "nest", "furniture", "modern", "contemporary"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Cloud Haven Chair",
    "description": "Sink into comfort with the Cloud Haven Chair—where softness meets support in a beautifully designed piece that feels like a personal retreat. Inspired by the gentle, calming presence of clouds, this chair offers an ultra-comfortable, inviting experience that lets you relax and unwind, making it the perfect addition to any living room, bedroom, or cozy nook. The Cloud Haven Chair is crafted with premium materials that create a plush, cloud-like seating experience. Its luxurious cushioning provides gentle support, cradling you in comfort while the ergonomic design ensures the perfect balance between relaxation and posture. The chair’s sleek, modern lines and soft, neutral tones blend seamlessly into any decor, adding a touch of sophistication and tranquility to your space. Whether you’re enjoying a good book, meditating, or simply relaxing after a long day, the Cloud Haven Chair creates a peaceful sanctuary where you can escape into serenity. Its durable construction guarantees lasting comfort and style, while its cloud-inspired design brings a sense of lightness and ease to your home. Key Features: Ultra-soft, cloud-like cushioning for a relaxing seating experience Ergonomically designed for optimal comfort and support Sleek, modern design that complements a variety of home decor styles Crafted from high-quality materials for durability and long-lasting comfort Perfect for creating a peaceful, inviting space in any room Transform your living space into a serene haven with the Cloud Haven Chair—where comfort and style come together to create the ultimate relaxation experience.",
    "discounted_price": 230,
    "original_price": 276.00,
    "tags": ["cloud", "chair", "comfy", "home decor", "modern furniture"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Bed",
    "description": "Introducing the Bed—your sanctuary for rest and relaxation, designed with both comfort and style in mind. This timeless piece is crafted to transform your bedroom into a peaceful retreat, offering a perfect balance of support, elegance, and durability. Whether you're outfitting a master bedroom or a guest room, the Bed ensures that every night is filled with restful sleep and every morning starts with ease. Constructed from high-quality materials, the Bed provides both sturdy support and luxurious comfort. The frame features sleek lines and a minimalist design, making it an ideal choice for a variety of interior styles, from modern to traditional. The soft yet supportive surface ensures that you can unwind in ultimate comfort, while the sturdy foundation provides long-lasting durability. Designed for those who appreciate both aesthetics and practicality, the Bed is a versatile addition to any bedroom. Its clean design, quality craftsmanship, and functional features create an inviting space where you can rest, recharge, and enjoy your most restful nights. Key Features: High-quality construction for durability and long-lasting support Sleek, minimalist design that fits into a wide range of bedroom styles Comfortable and supportive surface for a restful night's sleep Perfect for master bedrooms, guest rooms, or smaller spaces Built with attention to detail for both beauty and practicality Transform your bedroom into a restful haven with the Bed—where style, comfort, and functionality meet to create the ultimate space for relaxation and rejuvenation.",
    "discounted_price": 250,
    "original_price": 300.00,
    "tags": ["bed", "furniture", "sleep", "cozy", "modern"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Wood Chair",
    "description": "Introducing the Wood Chair—a beautifully crafted piece that blends timeless simplicity with natural elegance. Designed to bring warmth and character to any room, the Wood Chair is perfect for those who appreciate the beauty of wood and the durability of a well-made, functional chair. Whether you're adding a seating option to your dining room, office, or living area, this chair offers both style and comfort. Crafted from high-quality, sustainably sourced wood, the Wood Chair showcases the natural grain and texture of the wood, making each piece unique. Its clean lines and minimalist design highlight the inherent beauty of the material, allowing it to seamlessly complement a wide range of decor styles, from rustic and traditional to modern and contemporary. Not just a piece of furniture, the Wood Chair is built for comfort and practicality. Its sturdy construction provides reliable support, while the simple yet stylish design ensures it remains a staple in your home for years to come. Key Features: Made from high-quality, sustainably sourced wood for durability and style Clean, minimalist design that complements a variety of decor styles Comfortable seating and sturdy construction for everyday use Natural wood grain and texture make each chair unique Versatile design perfect for dining rooms, offices, or casual seating areas Add a touch of natural elegance to your space with the Wood Chair—where timeless design meets functionality for a comfortable and stylish seating experience.",
    "discounted_price": 100,
    "original_price": 120.00,
    "tags": ["wood", "chair", "furniture", "classic", "rustic"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Zen Table",
    "description": "Introduce tranquility and balance into your home with ZenTable—a beautifully crafted piece designed to inspire mindfulness, serenity, and effortless elegance. Inspired by the peaceful aesthetics of Zen design, this table serves as a functional centerpiece while fostering a calm, meditative atmosphere in any room. Perfect for modern homes or spaces dedicated to relaxation, ZenTable brings a sense of harmony and simplicity that enhances your daily life. Crafted with clean lines, natural materials, and subtle detailing, ZenTable captures the essence of minimalist beauty. Its simple yet striking design allows it to seamlessly integrate into a variety of interiors, whether placed in the living room, office, or meditation space. The ZenTable is built with both form and function in mind—offering a sturdy, durable surface while maintaining an elegant and soothing appearance. Whether used as a coffee table, a focal point in your meditation room, or a sleek accent in your living space, ZenTable encourages a peaceful, balanced environment. Its understated design creates an inviting atmosphere where you can reflect, unwind, and enjoy life's simpler moments. Key Features: Minimalist, Zen-inspired design that fosters peace and tranquility Crafted from high-quality natural materials for durability and elegance Simple, clean lines that complement modern and traditional interiors Versatile use in living rooms, offices, or meditation spaces Encourages balance and serenity, perfect for mindfulness practices Create a calming and harmonious space with ZenTable—where design meets tranquility, making every moment of your day more peaceful and centered.",
    "discounted_price": 250,
    "original_price": 300.00,
    "tags": ["zen", "table", "furniture", "calm", "minimalistic"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Timeless Elegance",
    "description": "Introducing TimelessElegance—a collection that embodies the perfect fusion of classic beauty and modern sophistication. Designed for those who appreciate the enduring appeal of refined style, TimelessElegance brings grace, charm, and unparalleled quality to any space. Each piece in this collection is crafted to stand the test of time, offering not only lasting durability but also a sense of sophistication that never goes out of style. With its clean lines, luxurious materials, and subtle detailing, TimelessElegance seamlessly complements any décor, from contemporary urban apartments to traditional homes. Whether you’re adding a statement piece to your living room, bedroom, or office, this collection brings an air of refinement that enhances every environment. The timeless design ensures that your space remains stylish for years to come, while its elegance offers a sense of warmth and comfort. Crafted with exceptional attention to detail, TimelessElegance pieces are built to withstand the rigors of daily life while retaining their beauty. Perfect for those who seek a blend of classic style and modern functionality, this collection is ideal for creating spaces that exude sophistication, grace, and enduring appeal. Key Features: Classic, timeless designs that offer long-lasting appeal Luxurious materials and meticulous craftsmanship for durability Versatile style that complements a wide range of home décor themes Perfect for creating a sophisticated, elegant atmosphere in any room Ideal for those who value style, quality, and refinement Add a touch of sophistication to your home with TimelessElegance—where grace meets functionality, and every detail is crafted to last a lifetime.",
    "discounted_price": 320,
    "original_price": 384.00,
    "tags": ["timeless", "elegance", "furniture", "classic", "luxury"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Sunny Chic",
    "description": "Embrace the warmth of style with SunnyChic—a vibrant and contemporary collection designed to bring the cheerful essence of sunshine and chic elegance to your home. Whether you're brightening up a living room, bedroom, or outdoor space, SunnyChic infuses every corner with a refreshing burst of energy and a touch of sophisticated charm. Inspired by the warmth of sunny days and the laid-back yet stylish vibe of coastal living, SunnyChic features bold colors, light fabrics, and breezy designs that capture the spirit of summer all year round. From sunny yellows and soft neutrals to playful patterns and textures, this collection effortlessly combines comfort with trendsetting design, creating spaces that feel both lively and inviting. Crafted from high-quality materials, SunnyChic is designed for those who appreciate a modern and cheerful aesthetic without sacrificing comfort. The collection offers a perfect balance of casual elegance and bright appeal, making it an ideal choice for those who love to incorporate light, airy tones and a touch of playfulness into their decor. Key Features: Bright, bold colors and breezy designs inspired by sunny, coastal living High-quality materials that are both durable and comfortable A versatile collection perfect for living rooms, bedrooms, or outdoor spaces Combines chic style with a relaxed, welcoming atmosphere Ideal for those looking to add a refreshing, cheerful touch to their home Fill your home with the light and energy of SunnyChic—where sunshine meets style for a lively and sophisticated living experience. Make every day feel like a sunlit escape.",
    "discounted_price": 400,
    "original_price": 480.00,
    "tags": ["sunny", "chic", "modern", "elegant", "furniture"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Timber Craft",
    "description": "Introducing TimberCraft—a collection that celebrates the timeless beauty of wood craftsmanship and the art of nature’s finest materials. Inspired by the rustic charm and durability of natural wood, TimberCraft brings warmth, character, and a touch of handcrafted elegance to any space. Perfect for those who value authenticity and sustainability, this collection combines the strength of timber with refined design, making it the ideal choice for modern and traditional homes alike. Each piece in the TimberCraft collection is meticulously crafted to highlight the unique grains and textures of the wood, bringing an organic, earthy feel to your space. Whether you're looking for furniture, decor, or accents, TimberCraft offers a variety of beautifully designed pieces that blend rustic appeal with contemporary sophistication. The collection is designed to stand the test of time, with materials chosen for their durability and lasting beauty. TimberCraft is perfect for those who appreciate quality, craftsmanship, and the enduring beauty of wood. From striking tables and chairs to decorative accessories, this collection adds a natural elegance to any room, transforming your living area into a warm and welcoming retreat. Key Features: Made from high-quality, sustainable timber for durability and lasting appeal Rustic yet refined design that complements both modern and traditional interiors Unique wood grains and textures that bring natural beauty to your space Expert craftsmanship that highlights the authenticity of each piece Perfect for creating a cozy, earthy atmosphere in any room Bring the timeless charm of nature into your home with TimberCraft—where the beauty of wood meets masterful craftsmanship, creating lasting pieces that stand out in both style and substance.",
    "discounted_price": 320,
    "original_price": 384.00,
    "tags": ["wooden", "craftsmanship", "furniture", "modern", "nature inspired"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Bright Space",
    "description": "Welcome to BrightSpace—a collection designed to infuse your home with light, energy, and vibrancy. Inspired by the power of natural light and open spaces, BrightSpace brings a fresh, airy feel to any room. Whether you're looking to brighten your living area, create an inspiring workspace, or transform your bedroom into a peaceful retreat, BrightSpace offers the perfect blend of style and function to make your environment feel open and inviting. Featuring clean lines, light hues, and functional designs, BrightSpace is all about creating a positive, uplifting atmosphere. Each piece is carefully crafted to enhance the flow of light, with materials that reflect brightness and encourage a sense of openness. The collection brings a modern, minimalist approach that allows your space to feel more expansive, whether through large windows, reflective surfaces, or light furniture choices. Ideal for those who appreciate a modern, fresh aesthetic with a focus on natural light, BrightSpace is perfect for those looking to create a lively yet serene environment. Whether you’re decorating a home office, living room, or sun-drenched dining area, this collection is designed to bring clarity and vibrancy to any space. Key Features: Designed to enhance natural light and create an open, airy feel Clean, minimalist lines that complement modern spaces Light, neutral tones and reflective surfaces for a bright atmosphere Crafted with quality materials for durability and lasting appeal Perfect for creating a vibrant, positive space in your home or office Transform your home into a bright, inviting haven with BrightSpace—where light, style, and functionality come together to create the perfect environment.",
    "discounted_price": 180,
    "original_price": 216.00,
    "tags": ["bright", "space", "minimalistic", "modern", "decor"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "The Dandy Chair",
    "description": "Meet The Dandy Chair—the epitome of comfort, style, and sophistication. Designed for those who appreciate the finer things in life, this chair combines classic charm with modern elegance, making it the perfect addition to any room in your home or office. Whether you're looking to create a luxurious reading nook, an inviting lounge area, or a statement piece for your workspace, The Dandy Chair offers both beauty and functionality. Crafted with meticulous attention to detail, The Dandy Chair features plush cushioning, ergonomic support, and a sleek, timeless design. Its high-quality materials and superior craftsmanship ensure durability while providing an exceptional seating experience. The chair's clean lines and refined shape give it a stylish yet relaxed feel, while its bold yet understated presence adds a touch of refinement to any space. Whether you choose a neutral tone or a pop of color, The Dandy Chair effortlessly complements a wide variety of decor styles, from modern minimalism to traditional luxury. Its versatility and comfort make it ideal for those who value both aesthetics and practicality in their furniture choices. Key Features: Comfortable, plush cushioning with ergonomic support Timeless design that adds sophistication to any space Crafted with high-quality materials for durability and longevity Versatile style that complements both modern and traditional interiors Ideal for living rooms, offices, or any area that calls for a touch of elegance Experience the perfect blend of comfort and style with The Dandy Chair—where luxury meets functionality and every detail is crafted to impress.",
    "discounted_price": 150,
    "original_price": 180.00,
    "tags": ["chair", "elegant", "vintage", "classic", "furniture"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Syltherine",
    "description": "Introducing Syltherine – the ultimate fusion of elegance and power. Crafted for those who demand exceptional performance, Syltherine is a premium product designed to elevate your experience to the next level. Whether you're seeking innovation, beauty, or unmatched durability, Syltherine offers all this and more. This cutting-edge solution stands at the intersection of advanced technology and refined aesthetics. Its sleek design, combined with carefully selected materials, promises not only a visually stunning appearance but also long-lasting reliability. Whether you are using Syltherine for professional tasks or personal indulgence, it delivers with precision, speed, and unmatched performance. Perfectly balanced to meet the needs of modern users, Syltherine is intuitive and easy to use, providing a seamless experience from start to finish. It boasts a variety of features to ensure that you stay ahead of the curve, whether it’s smart integration, energy efficiency, or superior functionality. Experience the future of performance with Syltherine—where luxury meets innovation, and every detail is engineered for perfection. Make Syltherine your go-to choice for an unparalleled experience today. Key Features: Superior performance and precision Sleek, modern design Built for durability and long-lasting use Easy integration and seamless user experience Ideal for both professionals and personal use Elevate your standards with Syltherine—where every moment is enhanced by innovation.",
    "discounted_price": 200,
    "original_price": 240.00,
    "tags": ["living", "fancy", "elegance", "desgin"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Marble Ease",
    "description": "Introducing MarbleEase—a luxurious collection that brings the timeless elegance of marble into your home with ease and sophistication. Designed for those who appreciate understated beauty and high-end design, MarbleEase combines the natural allure of marble with modern functionality, creating an effortless balance between style and practicality. Each piece in the MarbleEase collection features the exquisite veining and refined textures that make marble a classic choice, while offering lightweight and durable alternatives that ensure easy maintenance and long-lasting appeal. Whether you are adding a statement piece to your living room, kitchen, or office, MarbleEase brings a touch of opulence and tranquility to any space. From sleek tabletops and chic home accessories to sophisticated decor accents, MarbleEase infuses your home with a sense of luxury and simplicity. Its neutral tones and classic design complement a wide variety of interior styles, from modern minimalism to more traditional settings. Perfect for those who value timeless design with a contemporary twist, MarbleEase offers elegance without the hassle. Key Features: Timeless marble-inspired design with beautiful veining and texture High-quality, easy-to-care-for materials that offer durability and longevity Lightweight and functional, making it ideal for daily use and effortless maintenance Versatile pieces that complement a wide range of interior styles Perfect for adding an elegant, luxurious touch to any room Elevate your home with MarbleEase—where luxury meets ease, and timeless design is made effortlessly modern. Create a space that reflects your love for beauty, simplicity, and sophistication.",
    "discounted_price": 419,
    "original_price": 502.80,
    "tags": ["marble", "luxury", "furniture", "modern", "elegance"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Retro Vibe",
    "description": "Introducing RetroVibe—a perfect blend of vintage charm and modern style, designed for those who appreciate timeless aesthetics with a contemporary twist. Inspired by the bold designs and vibrant colors of the past, RetroVibe brings a nostalgic yet fresh vibe to any space, infusing your home with character and flair. Crafted with meticulous attention to detail, RetroVibe combines classic materials and distinctive design elements to create a standout piece that enhances your living space. From its sleek lines to its unique color palette, every aspect of RetroVibe is carefully crafted to evoke the spirit of retro design while seamlessly fitting into today’s modern interiors. Whether you’re decorating a living room, bedroom, or workspace, RetroVibe serves as a conversation starter and a statement piece. Its versatility allows it to complement a variety of decor styles, from mid-century modern to eclectic and contemporary. The perfect way to add a touch of nostalgia and style, RetroVibe is ideal for those who love vintage aesthetics with a modern twist. Key Features: Retro-inspired design with modern touches Bold color palette and classic materials that evoke nostalgia High-quality craftsmanship for durability and lasting appeal Versatile style that complements various home decor themes Ideal for creating a fun, stylish, and unique atmosphere Bring back the charm of the past with RetroVibe—where classic design meets contemporary flair. Perfect for those who love a dash of nostalgia in their modern home.",
    "discounted_price": 340,
    "original_price": 408.00,
    "tags": ["retro", "vintage", "furniture", "modern", "decor"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "The Lucky Lamp",
    "description": "Introducing The Lucky Lamp—a unique blend of charm, elegance, and positive energy designed to illuminate your home and bring good fortune into your life. This beautifully crafted lamp isn’t just a light source, it’s a symbol of prosperity and warmth, making it the perfect addition to any space that needs a touch of light and positivity. The Lucky Lamp features a stunning design that combines modern aesthetics with a timeless appeal. Its soft glow creates a calming atmosphere, while its symbolic design is said to attract good luck, success, and harmony. Whether placed on a bedside table, desk, or living room console, The Lucky Lamp adds a touch of magic and tranquility to your surroundings. Crafted with high-quality materials and attention to detail, The Lucky Lamp is not only a functional lighting solution but also a decorative piece that enhances your space. Its versatile design complements various decor styles, from contemporary to traditional, making it an ideal fit for any room in your home or office. Perfect for gifting or adding a touch of positivity to your own home, The Lucky Lamp is a thoughtful way to spread good energy and light wherever it goes. Key Features: Unique design that symbolizes good luck, prosperity, and harmony Soft, warm light that creates a calming atmosphere High-quality materials for durability and lasting appeal Versatile and stylish design that complements any decor Ideal for gifting or personal use Bring good fortune and illumination into your life with The Lucky Lamp—where light meets positivity and timeless design.",
    "discounted_price": 200,
    "original_price": 240.00,
    "tags": ["lamp", "lucky", "decor", "lighting", "vintage"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Pure Aura",
    "description": "Experience the essence of tranquility and purity with PureAura—a collection designed to bring peace, balance, and sophistication to your living space. Inspired by the clarity of nature and the calm of serene environments, PureAura transforms any room into a haven of peaceful elegance, creating an atmosphere that invites relaxation and mindfulness. Each piece in the PureAura collection is thoughtfully crafted with an emphasis on minimalism and clean design, allowing you to create a calm, clutter-free environment. The collection features soft, neutral tones, gentle textures, and natural materials, all chosen to enhance the serenity and flow of your home. Whether you're furnishing your living room, bedroom, or meditation space, PureAura offers a subtle yet powerful presence that promotes well-being and inner peace. Crafted from high-quality materials, every item in the PureAura collection is designed for both beauty and durability. From soothing furniture to delicate decorative accents, PureAura creates a harmonious balance between style and simplicity, making it the perfect choice for anyone who values a peaceful, refined space. Key Features: Minimalist design with soft, neutral tones for a calming atmosphere Crafted from natural materials that enhance the sense of tranquility High-quality craftsmanship for lasting beauty and durability Perfect for creating a peaceful, serene home environment Ideal for meditation rooms, bedrooms, or any space that requires relaxation and balance Elevate your home with PureAura—where simplicity meets elegance, creating an atmosphere of calm, clarity, and pure serenity. Let your space reflect the tranquility and balance that you deserve.",
    "discounted_price": 280,
    "original_price": 336.00,
    "tags": ["pure", "modern", "elegance", "interior design", "furniture"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Tropical Vibe",
    "description": "Escape to paradise with TropicalVibe—a collection that brings the vibrant energy of the tropics right into your home. Inspired by lush landscapes, warm sunshine, and vibrant colors, TropicalVibe offers a refreshing, tropical-inspired aesthetic that creates a relaxed and inviting atmosphere in any space. Designed for those who dream of a laid-back, vacation-like ambiance, TropicalVibe captures the essence of island living with its lively prints, natural textures, and bold colors. Whether you're looking to add a touch of the tropics to your living room, bedroom, or outdoor patio, TropicalVibe seamlessly infuses any area with a sense of adventure and tranquility. Crafted from high-quality materials, each piece in the TropicalVibe collection is both durable and stylish. The collection features elements such as leafy motifs, rattan finishes, and tropical florals that effortlessly capture the spirit of tropical destinations. Whether you’re lounging in your living room or hosting an outdoor gathering, TropicalVibe creates the perfect backdrop for relaxation and good vibes. Key Features: Tropical-inspired designs that evoke warmth and relaxation Lush colors, natural textures, and island-inspired prints High-quality materials for comfort, durability, and style Perfect for creating a vacation-like ambiance in any room or outdoor space Ideal for those who love bold, vibrant, and nature-infused decor Bring the beauty and energy of the tropics into your home with TropicalVibe—where island-inspired design meets everyday comfort and style. Let every day feel like a getaway.",
    "discounted_price": 550,
    "original_price": 660.00,
    "tags": ["tropical", "vibe", "furniture", "exotic", "decor"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Modern Serenity",
    "description": "Introducing ModernSerenity—a collection that redefines contemporary living by combining tranquility and style in perfect harmony. Designed for those who seek balance, simplicity, and elegance, ModernSerenity transforms any space into a peaceful retreat, offering a serene atmosphere that promotes relaxation and well-being. The essence of ModernSerenity lies in its minimalist approach, where clean lines, neutral tones, and soothing textures create an environment that invites calmness and clarity. Whether you're furnishing your living room, bedroom, or office, each piece in the ModernSerenity collection is designed to elevate your space while maintaining an effortless, uncluttered aesthetic. Crafted with the finest materials, ModernSerenity products are built to last and are designed with both beauty and practicality in mind. From soft, inviting fabrics to sleek, functional shapes, every detail has been thoughtfully considered to enhance your lifestyle. With a focus on smart design, ModernSerenity delivers a timeless sense of peace and elegance, making it the ideal choice for anyone looking to create a sophisticated, serene home. Key Features: Sleek, minimalist design for a calm and tranquil environment High-quality materials for durability and comfort Neutral tones and soft textures that complement any decor Perfect for creating a serene, modern living space Ideal for individuals seeking balance and relaxation in their home or office Transform your space with ModernSerenity—where contemporary design meets peaceful living. Create a home that reflects your sense of style and inner calm.",
    "discounted_price": 480,
    "original_price": 576.00,
    "tags": ["modern", "serenity", "peaceful", "contemporary", "furniture"],
    "add_to_cart": "Add To Cart"
  },
  {
    "name": "Reflective Haven",
    "description": "Create a sanctuary of calm and introspection with ReflectiveHaven—a collection designed to bring peace, tranquility, and thoughtful elegance into your space. Inspired by serene environments and mindful living, ReflectiveHaven is perfect for those seeking a retreat from the hustle and bustle of everyday life. This collection invites you to slow down, reflect, and enjoy moments of stillness. The pieces in the ReflectiveHaven collection are designed with clean lines, soothing tones, and a harmonious blend of textures that foster a sense of balance and serenity. From minimalist furniture to subtle decorative accents, each item is thoughtfully crafted to create an atmosphere that encourages mindfulness and personal reflection. The calming neutral hues, natural materials, and understated beauty make it easy to turn any room into a peaceful haven. Perfect for meditation spaces, cozy reading nooks, or tranquil bedrooms, ReflectiveHaven brings a sense of clarity and inner peace to your home. It’s for those who value simplicity, balance, and the ability to create an environment that nurtures both mental clarity and physical relaxation. Key Features: Minimalist design with a focus on calm, neutral tones and natural materials High-quality craftsmanship for lasting beauty and durability Perfect for creating peaceful, reflective spaces within your home Ideal for meditation, relaxation, or simply unwinding after a long day Invites mindfulness and inner peace with every detail Transform your living space into a peaceful retreat with ReflectiveHaven—where serene design meets mindful living, creating a sanctuary for reflection and relaxation.",
    "discounted_price": 300,
    "original_price": 360.00,
    "tags": ["reflective", "haven", "contemporary", "modern", "decor"],
    "add_to_cart": "Add To Cart"
  }
]

@function_tool
def get_updated_website(query: str) -> str:
    """
    Function to get the latest website updates.
    """
    print(f"Received query: {query}")
    return """
    Furniro E-Commerce Website Overview

    The Furniro site (at final-project-ecommerce-website.vercel.app) is a furniture and home-decor online store. Its homepage and pages use a clean template branded “Furniro” with product examples like chairs, sofas, tables, lamps, mugs, bedding, and plant pots. For example, the homepage lists items named Syltherine Stylish cafe chair, Lolito Luxury big sofa, Respira Outdoor bar table and stool, Grifo Night lamp, Muggo Small mug, Pingky Cute bed set, Potty Minimalist flower pot, etc. (shown with Indonesian-style prices, “Rp”
    final-project-ecommerce-website.vercel.app
). The site clearly focuses on home/furnishing products (dining, living, bedroom furniture and decor).

Homepage Sections

Navigation Bar (Header): At the top is the Furniro logo and a main menu with links Home, Shop, Blog, Contact
final-project-ecommerce-website.vercel.app
. Clicking the logo or “Home” returns to the homepage; “Shop” goes to the product listing page; “Blog” goes to the blog; “Contact” goes to the contact form page.

Hero / New Collection Banner: Below the header is a “New Arrival” banner. It displays a big title “Discover Our New Collection”, some placeholder text (Lorem ipsum), and a BUY NOW button
final-project-ecommerce-website.vercel.app
. This is the main promotional callout. (In our site scrape it appears as static text and the “BUY NOW” looks like a button or link, presumably intended to link to featured products or the Shop page, though it does not function in the scraped HTML.)

“Browse The Range” Section: Next is a section titled “Browse The Range”. It shows three large images labeled Dining, Living, and Bedroom
final-project-ecommerce-website.vercel.app
final-project-ecommerce-website.vercel.app
. Each image caption (Dining/Living/Bedroom) is meant to represent a category of furniture. (On some templates these images could filter the Shop view by category, but on this site they seem decorative.) The section’s heading and the category labels are visible in the markup
final-project-ecommerce-website.vercel.app
.

Figure: Category showcase images for Dining (left), Living (center), and Bedroom (right), as seen in the “Browse The Range” section.

The text “Browse The Range” appears above these images
final-project-ecommerce-website.vercel.app
. This section invites users to explore by category. In the site’s code, the captions “Dining”, “Living”, “Bedroom” appear exactly under each image
final-project-ecommerce-website.vercel.app
.

“Our Products” Section: Further down is a Products showcase titled “Our Products”. It displays a grid of featured items. Each product tile shows a name, description, price, and tags like -50% or New. For example, one item is Syltherine – Stylish cafe chair priced “Rp 2.500.000” with original price “Rp 3.500.000” struck out (indicating a 50% discount)
final-project-ecommerce-website.vercel.app
. Other items (Lolito sofa, Respira stool, Grifo lamp, Muggo mug, etc.) are listed similarly in this section
final-project-ecommerce-website.vercel.app
. At the end of the grid is a “Show More” link
final-project-ecommerce-website.vercel.app
, which presumably would load or link to the full Shop page. (In the HTML we see the text “Show More” at the bottom of the product list
final-project-ecommerce-website.vercel.app
.)

“Beautiful rooms inspiration” Section: Below the products is a gallery section with title “50+ Beautiful rooms inspiration”
final-project-ecommerce-website.vercel.app
. It includes descriptive text (“Our designer already made a lot of beautiful prototype of rooms that inspire you”) and images. Two sample images (a dining table scene and a plant/bedroom scene) are shown, with captions like “01 Bed Room – Inner Peace”
final-project-ecommerce-website.vercel.app
. This section has an Explore More link for viewing more room inspiration
final-project-ecommerce-website.vercel.app
.

Figure: Sample inspiration images from the homepage (labeled “Inner Peace” and another styled room). The “Beautiful rooms inspiration” gallery invites users to browse designer room photos
final-project-ecommerce-website.vercel.app
.

Footer: At the bottom is a footer with the company name/address, site links, help links, and newsletter signup. It shows Funiro. with an address (Coral Gables, FL) and lists link sections. Under Links it repeats the main menu (Home, Shop, Blog, Contact)
final-project-ecommerce-website.vercel.app
. Under Help it lists “Payment Options”, “Returns”, “Privacy Policies” (these appear as text, likely intended to be links). There is also a “Newsletter” field with a SUBSCRIBE button
final-project-ecommerce-website.vercel.app
. (Finally a copyright “2023 furino. All rights reserved” is shown
final-project-ecommerce-website.vercel.app
.)

Shop Page

The Shop page (accessed by clicking the Shop link) is labeled “Shop” at the top with a breadcrumb “Home > Shop”
final-project-ecommerce-website.vercel.app
. Its main heading says “Product List”. In the visible HTML the page unfortunately has no actual products listed, but it does show feature highlights. Below the heading are four promo icons/text blocks reading High Quality (crafted from top materials), Warranty Protection (Over 2 years), Free Shipping (Order over 150 $), and 24/7 Support (Dedicated support)
final-project-ecommerce-website.vercel.app
final-project-ecommerce-website.vercel.app
. These are decorative assurances but no product grid appears in the static HTML. (Presumably a functional site would list all products here or allow filtering, but our view shows only the feature blurbs.) The footer (address, links, etc.) is the same as the homepage.

Blog Page

The Blog page is an articles listing. At top it says “Blog” with “Home > Blog” breadcrumb
final-project-ecommerce-website.vercel.app
. It shows multiple blog post previews in a two-column layout. Each post preview has an image, author/date (“Admin • 14 Oct 2022”), category (e.g. “Wood”), a title (e.g. Going all-in with millennial design), some dummy text, and a “Read More” label
final-project-ecommerce-website.vercel.app
. For example, the first post “Going all-in with millennial design” (wood category) is visible with its snippet and a “Read More” link
final-project-ecommerce-website.vercel.app
. There are three such posts shown on the page.

A sidebar lists Categories (Crafts 2, Design 8, Handmade 7, Interior 1, Wood 6)
final-project-ecommerce-website.vercel.app
 and Recent Posts (thumbnail+title+date for the three latest)
final-project-ecommerce-website.vercel.app
. The page also includes the same four feature blurbs (High Quality, Warranty, Free Shipping, 24/7 Support) and the same footer as other pages. All navigation links (header/footer) remain the same.

Contact Page

The Contact page (Home > Contact) is titled “Get In Touch With Us”. It presents company contact info and a form. On the left it shows the Address, Phone, and Working Time (Monday–Friday, Saturday–Sunday hours)
final-project-ecommerce-website.vercel.app
. For example, it lists an address in New York, phone numbers (“+(84) 546-6789” etc.), and working hours (9am–10pm on weekdays, 9–9 on weekends)
final-project-ecommerce-website.vercel.app
. On the right is a contact form with fields Your name, Email address, Subject, Message, and a Submit button
final-project-ecommerce-website.vercel.app
. (These appear in the HTML as text inputs with labels.) After the form, again the four feature blurbs and footer are shown. The header nav (Furniro logo and links) still appears at top.

Clickable Elements

All primary buttons and links on the site are as follows:

Logo/Home: The “Furniro” logo at top (and the “Home” link) always go back to the homepage.

Navigation Links: The header links “Shop”, “Blog”, and “Contact” go to the Shop, Blog, and Contact pages respectively
final-project-ecommerce-website.vercel.app
. (The same links are also repeated in the footer.)

BUY NOW button: In the homepage hero, the BUY NOW text appears as a button. In a complete site this would likely link to a featured product or the Shop, but here it does not have a distinct hyperlink in the static HTML.

Category images: The Dining/Living/Bedroom images in “Browse The Range” are not actual hyperlinks in the scraped HTML (clicking them in our test opened the image itself). However, in a full implementation they might filter the Shop view by category.

Product titles/images: In “Our Products”, the product names/images are listed but did not appear as separate links in the HTML. (In the site we inspected, clicking on the product block text did not yield a new page in our console view.) In many e-commerce layouts these would link to a detail page.

Add to Cart / Share / Compare / Like: In some templates (like the related “Furniro” templates found online), each product tile has an Add to cart button and share/compare/like icons. In our site HTML we see no separate add-to-cart buttons on the homepage (it looks like a simplified version). The related Furniro example [60] even shows “Add to cart” buttons for each product
furniro.ehasun.com
, but our site did not render those.

Show More: Under the product grid is a Show More link
final-project-ecommerce-website.vercel.app
. Clicking “Show More” should take the user to the full Shop/Product List page. (Indeed our console shows that text, which likely links to the Shop.)

Read More: Each blog post preview has “Read More” at the end
final-project-ecommerce-website.vercel.app
. That would normally link to the full blog article. In our scrape it is just text, but the structure indicates a link is intended.

Explore More: In the inspiration section, “Explore More” is shown
final-project-ecommerce-website.vercel.app
, presumably linking to a gallery or blogpost about room ideas.

Footer Links: “Payment Options”, “Returns”, “Privacy Policies” are listed under Help in the footer
final-project-ecommerce-website.vercel.app
. These likely should link to informational pages, though in our HTML they appear as plain text.

Form Submit: On the Contact page, the Submit button sends the form (though in our static view it does nothing visible; normally it would attempt to email or otherwise send the message).

Browsing and Purchase Flow

In a typical use of this site, a customer would:

Find Products: Click the Shop link (or the homepage’s “Show More”) to view all products. (Alternatively, one might browse the homepage “Our Products” section, but to see everything, the Shop page is intended.) The Shop page lists products with prices.

Select an Item: Click on a product to view its details. (This step is implied by the design, though our inspection didn’t reveal separate product-detail pages. On many e-commerce sites, clicking a product name or image goes to a product page.)

Add to Cart: On the product page, the user would choose any options (size, color) and click Add to cart. The site’s structure suggests an Add-to-cart function exists (the related Furniro example shows “Add to cart” buttons
furniro.ehasun.com
), but our scrape did not expose the actual cart.

View Cart: After adding items, the user would click a cart icon or link (not visibly present in the homepage HTML) to review their shopping cart. The cart page would list chosen products, quantities, and total cost.

Checkout: From the cart, the user would proceed to checkout, enter shipping/payment information, and place the order.

Note: In the accessible source code, we did not find any visible cart or checkout pages. Therefore the above steps describe the expected flow, but the site did not actually display a cart interface in our view. It appears to be a template for an e-commerce flow.

Navigating to Specific Categories or Products

To go directly to a specific category or product, a user has these options:

Using the Header Menu: The Shop link leads to the full product listing (all categories)
final-project-ecommerce-website.vercel.app
. From there, if implemented, one could filter or search by category (though our static view has no visible filter UI).

From Homepage Categories: The “Dining”, “Living”, “Bedroom” images on the homepage suggest major categories
final-project-ecommerce-website.vercel.app
. In some implementations, clicking those images would filter the Shop page to only show that category. In our site these images are present (with captions) but they do not have separate hyperlinks in the scraped HTML. They serve as visual cues for the category sections.

“Show More” Link: In the homepage’s products section, clicking Show More
final-project-ecommerce-website.vercel.app
 takes the user to the Shop page (effectively the same as the Shop link in the nav).

Blog Categories (for blog content): On the blog page, clicking on a category name (e.g. “Wood”) would filter posts, but this only affects blog content, not products.

Direct URL: A user could also type the URL of a known product or category page if known (e.g. final-project-ecommerce-website.vercel.app/Shop), but the site is minimal on interlinking.

In summary, the main navigation to products is via the Shop link or the homepage product links. The homepage categories and “Show More” provide clues to narrow down selections, but in the current version there is no interactive search or filter interface visible
    """

@function_tool
def get_update_payment(query: str) -> str:
    """
    Function to get the latest payment updates.
    """
    print(f"Received query: {query}")
    return f"The latest payment updates include new payment methods, enhanced security features, and improved transaction processing times."

# Products Agent
products_agent = Agent(
    name="Products_Agent",
    instructions="You are a specialized agent for product-related all queries. You can provide information about products, their features, price of each product specifications and all information did you have.",
    model=model,
    tools=[get_updated_products],  # Register the function tool and product info tool
)

# Website Overview Agent
website_overview_agent = Agent(
    name="Website_Overview_Agent",
    instructions="You are a specialized agent for website overview queries. You can provide all website information, all overview detail and all information did you have.",
    model=model,
    tools=[get_updated_website],  # Register the function tool
)

# Payment Agent
payment_agent = Agent(
    name="Payment_Agent",
    instructions="You are a specialized agent for payment-related queries. You can provide information about payment methods, payment status, and any payment-related issues.",
    model=model,
    tools=[get_update_payment],  # Register the function tool
    
)


# AI Chatbot Orschestrator Agent
ai_chatbot_agent = Agent(
    name = "AI_Chatbot_Agent",
    instructions = "You are a helpful AI chatbot assistant designed to answer user questions and resolve their queries regarding a website, products, and payments. Your Role: Website: Provide comprehensive details about the website, including its services, ratings, and benefits. Use tools to resolve any website-related issues. Products: Give information on product descriptions, features, pricing, and stock availability. You can also help resolve any product-related issues. Payments: Offer information on payment methods, payment status, and help with any payment-related issues.",
    model=model,
    tools = [
        products_agent.as_tool(
            tool_name="Products_Agent",
            tool_description="You are a specialized agent for product-related queries. You can provide information about products, their features, price of each product and specifications.",
        ),
        website_overview_agent.as_tool(
            tool_name="Website_Overview_Agent",
            tool_description="You are a specialized agent for website overview queries. You can provide all website information, all overview and details.",
        ),
        payment_agent.as_tool(
            tool_name="Payment_Agent",
            tool_description="You are a specialized agent for payment-related queries. You can provide information about payment methods, payment status, and any payment-related issues.",
        ),
    ],
)

class ChatMessage(BaseModel):
    message: str


@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    """
    FastAPI endpoint to handle chat messages.
    """
    try:
        result = await Runner.run(ai_chatbot_agent, chat_message.message, run_config=run_config)
        return {"response": result.final_output}
        print(f"Response: {result.final_output}")
    except Exception as e:
        return {"error": str(e)}


# @cl.on_message
# async def handle_message(message: cl.Message):
#     """
#     Chainlit message handler to process incoming messages.
#     """
#     result = await Runner.run(ai_chatbot_agent, message.content, run_config=run_config)
#     await cl.Message(content=f"Response: {result.final_output}").send()
#     print(f"Response: {result.final_output}")


