===AIMED===

## Opis

AI MED je projekt usmjeren na automatsku transkripciju liječničkih dijagnoza iz govora u tekst. Cilj je ubrzati proces pisanja i oblikovanja medicinskih nalaza pomoću umjetne inteligencije, kako bi liječnici mogli brže i učinkovitije izdavati precizne dijagnoze svojim pacijentima.

## Struktura 

src/ -> tu se nalazi glavni program AIMED.py
docs/ -> projektna dokumentacija
data/prompts -> promptovi koristeni u prompt biasu  
data/dataset -> dataset iz medicinskog korpusa

## Zahtjevi sustava - Windows

- Python 3.13+
- pip
- ffmpeg

##Zahtjevi sustava - Linux

Obuhvaća zahtjeve sustava za Windows te korištenje drugih tkinter biblioteka za GUI, no ako to nije izvedivo koristiti browser inačicu u src/ai2.python
 
## Instalacija

git clone https://github.com/Ivor01/AIMED
cd AIMED
pip install -r requirements.txt

## Pokretanje
cd AIMED/src
python AIMED.py


## Whisper model

Za korištenje fine tuneane verzije Whisper modela potrebna je zasebno preuzimanje modela zbog njegove veličine.
