![zxdfdsxf.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/EnlRwJl06-OKBOYm7l0Lh.png)

# **Food-101-93M**

> **Food-101-93M** is a fine-tuned image classification model built on top of **google/siglip2-base-patch16-224** using the **SiglipForImageClassification** architecture. It is trained to classify food images into one of 101 popular dishes, derived from the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

```py
Classification Report:
                         precision    recall  f1-score   support

              apple_pie     0.8399    0.8253    0.8325       750
         baby_back_ribs     0.9445    0.8853    0.9140       750
                baklava     0.9736    0.9347    0.9537       750
         beef_carpaccio     0.9079    0.9200    0.9139       750
           beef_tartare     0.8486    0.8293    0.8388       750
             beet_salad     0.8649    0.8707    0.8678       750
               beignets     0.8961    0.9080    0.9020       750
               bibimbap     0.9361    0.9373    0.9367       750
          bread_pudding     0.7979    0.8000    0.7989       750
      breakfast_burrito     0.8784    0.9053    0.8917       750
             bruschetta     0.8672    0.8533    0.8602       750
           caesar_salad     0.9444    0.9293    0.9368       750
                cannoli     0.9263    0.9547    0.9402       750
          caprese_salad     0.9110    0.9280    0.9194       750
            carrot_cake     0.9068    0.8040    0.8523       750
                ceviche     0.8375    0.8453    0.8414       750
             cheesecake     0.8225    0.8093    0.8159       750
           cheese_plate     0.9627    0.9627    0.9627       750
          chicken_curry     0.8970    0.8827    0.8898       750
     chicken_quesadilla     0.9254    0.9093    0.9173       750
          chicken_wings     0.9512    0.9360    0.9435       750
         chocolate_cake     0.7958    0.8107    0.8032       750
       chocolate_mousse     0.6947    0.7827    0.7361       750
                churros     0.9440    0.9440    0.9440       750
           clam_chowder     0.8883    0.9120    0.9000       750
          club_sandwich     0.9396    0.9133    0.9263       750
             crab_cakes     0.9185    0.8720    0.8947       750
           creme_brulee     0.9141    0.9227    0.9184       750
          croque_madame     0.9106    0.8960    0.9032       750
              cup_cakes     0.8986    0.9333    0.9156       750
           deviled_eggs     0.9787    0.9813    0.9800       750
                 donuts     0.8893    0.8787    0.8840       750
              dumplings     0.9212    0.8880    0.9043       750
                edamame     0.9960    0.9920    0.9940       750
          eggs_benedict     0.9207    0.9440    0.9322       750
              escargots     0.8709    0.8907    0.8807       750
                falafel     0.8945    0.8933    0.8939       750
           filet_mignon     0.7598    0.7467    0.7532       750
         fish_and_chips     0.9454    0.9467    0.9460       750
              foie_gras     0.6659    0.8027    0.7279       750
           french_fries     0.9447    0.9333    0.9390       750
      french_onion_soup     0.8667    0.9187    0.8919       750
           french_toast     0.8890    0.8760    0.8825       750
         fried_calamari     0.9448    0.9133    0.9288       750
             fried_rice     0.9325    0.9213    0.9269       750
          frozen_yogurt     0.8716    0.9507    0.9094       750
           garlic_bread     0.9103    0.8800    0.8949       750
                gnocchi     0.8554    0.8280    0.8415       750
            greek_salad     0.9203    0.9240    0.9222       750
grilled_cheese_sandwich     0.8523    0.8773    0.8647       750
         grilled_salmon     0.8463    0.8960    0.8705       750
              guacamole     0.9537    0.9347    0.9441       750
                  gyoza     0.8970    0.9173    0.9071       750
              hamburger     0.8899    0.8947    0.8923       750
      hot_and_sour_soup     0.9439    0.9413    0.9426       750
                hot_dog     0.8859    0.9320    0.9084       750
       huevos_rancheros     0.8465    0.8827    0.8642       750
                 hummus     0.9394    0.9093    0.9241       750
              ice_cream     0.8633    0.8507    0.8570       750
                lasagna     0.8780    0.8733    0.8757       750
         lobster_bisque     0.8952    0.9107    0.9028       750
  lobster_roll_sandwich     0.9664    0.9573    0.9618       750
    macaroni_and_cheese     0.9273    0.9013    0.9141       750
               macarons     0.9892    0.9747    0.9819       750
              miso_soup     0.9565    0.9667    0.9615       750
                mussels     0.9602    0.9640    0.9621       750
                 nachos     0.9337    0.9387    0.9362       750
               omelette     0.8889    0.8960    0.8924       750
            onion_rings     0.9493    0.9493    0.9493       750
                oysters     0.9808    0.9533    0.9669       750
               pad_thai     0.9188    0.9507    0.9345       750
                 paella     0.9352    0.9240    0.9296       750
               pancakes     0.9277    0.9067    0.9171       750
            panna_cotta     0.8056    0.8507    0.8275       750
            peking_duck     0.8529    0.9120    0.8814       750
                    pho     0.9746    0.9227    0.9479       750
                  pizza     0.9512    0.9360    0.9435       750
              pork_chop     0.8085    0.7373    0.7713       750
                poutine     0.9424    0.9387    0.9405       750
              prime_rib     0.9106    0.8147    0.8600       750
   pulled_pork_sandwich     0.8887    0.9053    0.8970       750
                  ramen     0.8986    0.9213    0.9098       750
                ravioli     0.8532    0.8293    0.8411       750
        red_velvet_cake     0.9330    0.8907    0.9113       750
                risotto     0.8809    0.8680    0.8744       750
                 samosa     0.9153    0.9227    0.9190       750
                sashimi     0.9248    0.9187    0.9217       750
               scallops     0.8564    0.8507    0.8535       750
          seaweed_salad     0.9597    0.9533    0.9565       750
       shrimp_and_grits     0.8995    0.8947    0.8971       750
    spaghetti_bolognese     0.9667    0.9667    0.9667       750
    spaghetti_carbonara     0.9601    0.9627    0.9614       750
           spring_rolls     0.9045    0.9467    0.9251       750
                  steak     0.6311    0.7027    0.6650       750
   strawberry_shortcake     0.8832    0.8467    0.8645       750
                  sushi     0.9204    0.8947    0.9074       750
                  tacos     0.9225    0.8893    0.9056       750
               takoyaki     0.9419    0.9507    0.9463       750
               tiramisu     0.9074    0.8627    0.8845       750
           tuna_tartare     0.7691    0.7773    0.7732       750
                waffles     0.9629    0.9347    0.9486       750

               accuracy                         0.8973     75750
              macro avg     0.8987    0.8973    0.8977     75750
           weighted avg     0.8987    0.8973    0.8977     75750
```

The model categorizes images into 101 food classes such as `sushi`, `hamburger`, `waffles`, `pad_thai`, and more.

---

# **Run with Transformers 🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Food-101-93M"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Food-101 labels
labels = {
    "0": "apple_pie", "1": "baby_back_ribs", "2": "baklava", "3": "beef_carpaccio", "4": "beef_tartare",
    "5": "beet_salad", "6": "beignets", "7": "bibimbap", "8": "bread_pudding", "9": "breakfast_burrito",
    "10": "bruschetta", "11": "caesar_salad", "12": "cannoli", "13": "caprese_salad", "14": "carrot_cake",
    "15": "ceviche", "16": "cheesecake", "17": "cheese_plate", "18": "chicken_curry", "19": "chicken_quesadilla",
    "20": "chicken_wings", "21": "chocolate_cake", "22": "chocolate_mousse", "23": "churros", "24": "clam_chowder",
    "25": "club_sandwich", "26": "crab_cakes", "27": "creme_brulee", "28": "croque_madame", "29": "cup_cakes",
    "30": "deviled_eggs", "31": "donuts", "32": "dumplings", "33": "edamame", "34": "eggs_benedict",
    "35": "escargots", "36": "falafel", "37": "filet_mignon", "38": "fish_and_chips", "39": "foie_gras",
    "40": "french_fries", "41": "french_onion_soup", "42": "french_toast", "43": "fried_calamari", "44": "fried_rice",
    "45": "frozen_yogurt", "46": "garlic_bread", "47": "gnocchi", "48": "greek_salad", "49": "grilled_cheese_sandwich",
    "50": "grilled_salmon", "51": "guacamole", "52": "gyoza", "53": "hamburger", "54": "hot_and_sour_soup",
    "55": "hot_dog", "56": "huevos_rancheros", "57": "hummus", "58": "ice_cream", "59": "lasagna",
    "60": "lobster_bisque", "61": "lobster_roll_sandwich", "62": "macaroni_and_cheese", "63": "macarons", "64": "miso_soup",
    "65": "mussels", "66": "nachos", "67": "omelette", "68": "onion_rings", "69": "oysters",
    "70": "pad_thai", "71": "paella", "72": "pancakes", "73": "panna_cotta", "74": "peking_duck",
    "75": "pho", "76": "pizza", "77": "pork_chop", "78": "poutine", "79": "prime_rib",
    "80": "pulled_pork_sandwich", "81": "ramen", "82": "ravioli", "83": "red_velvet_cake", "84": "risotto",
    "85": "samosa", "86": "sashimi", "87": "scallops", "88": "seaweed_salad", "89": "shrimp_and_grits",
    "90": "spaghetti_bolognese", "91": "spaghetti_carbonara", "92": "spring_rolls", "93": "steak", "94": "strawberry_shortcake",
    "95": "sushi", "96": "tacos", "97": "takoyaki", "98": "tiramisu", "99": "tuna_tartare", "100": "waffles"
}

def classify_food(image):
    """Predicts the type of food in the image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    # Sort by descending probability
    predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True)[:5])
    
    return predictions

# Gradio Interface
iface = gr.Interface(
    fn=classify_food,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=5, label="Top 5 Prediction Scores"),
    title="Food-101-93M 🍽️",
    description="Upload an image of food to classify it into one of 101 dish categories based on the Food-101 dataset."
)

# Launch app
if __name__ == "__main__":
    iface.launch()
```

---

# **Intended Use:**

The **Food-101-93M** model is intended for:

- **Recipe Recommendation Engines:** Automatically tagging food images to suggest recipes.
- **Food Logging & Calorie Tracking Apps:** Categorizing meals based on photos.
- **Smart Kitchens:** Assisting food recognition in smart appliances.
- **Restaurant Menu Digitization:** Auto-classifying dishes for visual menus or ordering systems.
- **Dataset Labeling:** Enabling automatic annotation of food datasets for training other ML models.
