# Description

- Introduccion
  - Document Objective
  - Glossary
  
- Protection Steps List

- Step Explanation
  - NLP NER extraction
  - Merge consecutive detection tokens into one detection
  - Remove void and noise detections
  - Merge consecutive ADDRESS detections
  - Extract manual tokens from the text
  - Translate detections to semantic terms.
  - Link detections to data properties using natural language terms.
  - Clean low information detections for names
  - Add detections using data patterns
  - Select the most restrictive security label per detection
  - Deduplicate detections
  - Ensure security label
  - Tag gaps between consecutive detections
  - Remove unwanted detections
  - Merge consecutive detections with the same security label
  
## Introduction
The developed system aims to protect client conversations with the bank human agents. In order to do so, the system relies on a NLP module and - in addition- it applies several steps to complement NLP. These complements use traditional pattern extraction logic in order to achieve the best protection.

### Document Objective
To describe the protection logic in order to help understand the system's behavior

### Glossary
This section briefly describes the terms used in the document:
  - **Token:** consecutive set of characters. Typically different tokens are separated by spaces.
  - **Entity:** a type/class that can be identified. E.g., “NAME”, “ADDRESS”
  - **Detection:** a set of characters that represent an entity classified by the system. E.g., “Peter Smith” as a NAME.
  - **Tokenize:** final hashing method to implement on the detection text. This term's meaning comes from BBVA usage.
  - **NER:** Named entity recognition. It is the NLP task of identifying entities from a text.
  - **NLP Label:** classification assigned to a detection. This label comes from the NER.
  - **Security Label:** classification for an entity that follows the Bank's definition. They come from RCN.
  - **Semantic Label:** classification of the detection according to its meaning. It is an intermediary label between “NLP Label” and “Security Label”.

## Protection Steps List
  - NLP NER extraction
  - Merge consecutive detection tokens into one detection
  - Remove void and noise detections
  - Merge consecutive ADDRESS detections
  - Extract manual tokens from the text.
  - Translate detections to semantic terms.
  - Link detections to data properties using natural language terms
  - Clean low information detections for names 
  - Add detections using data patterns.
  - Select the most restrictive security label per detection.
  - Deduplicate detections.
  - Ensure security label.
  - Tag gaps between consecutive detections.
  - Remove unwanted detections.
  - Merge the consecutive detections with the same security label.

## Step Explanation

Each one of the steps is briefly described in this section. Every step is an incremental procedure applied to the results of the previous one.

### NLP NER extration

This is the first step. This step delegates the full text to the trained NLP module. A set of detections is returned as the result of the NER task.

These detections will be complemented and processed in the following steps.

Every detection contains the original target text and the metadata that describes it in this context. The most relevant metadata fields are:

- **Start position:** absolute character index from the beginning of the conversation where the detection starts.
- **End position:** absolute character index from the beginning of the conversation where the detection ends.
- **Value:** the text that is detected.
- **NLP label:** the NLP label that the NER module guessed.
- **Detected type:** the semantic translation of the NLP label
- **Security label:** the security label that corresponds to the type of the detection.
- **Security class:** the security class of this security label.
- **Protection method:** how to protect this entity as provided by the enforcing system.
- **Tokenization Type:** how to 'tokenize' this entity as provided by the enforcing system.
- **Certainty:** percentage that represents how confident the system is about the classification of the detection.
- **Possible data properties:** the set of data properties in the ontology that could be types for the detection.

These fields are incrementally populated as the logic progresses through the logic steps described in this document.

### Merge consecutive detection tokens into one detection
The NER task provides classifications for every token. Consecutive tokens can be included in the same detection. This step joins consecutive tokens with the same classification into a single detection.

    E.g.,
    TEXT:                Peter Smith lives in Brazil.
    DETECTIONS FROM NER: NNNNN NNNNN OOOOO OO OOOOOO
    MERGED DETECTIONS:   NNNNNNNNNNN OOOOOOOOOOOOOOO
    (NOTE: 'N' stands for 'NAME' in this example)

### Remove void and noise detections
The detections so far are a product of the NLP NER task. The module classifies the not-relevant text as 'O' -other, or not relevant- and could also be classifying very small pieces of text .

This step removes the 'not relevant' classifications and the ones that are 2 or less characters long.

### Merge consecutive ADDRESS detections

The addresses are a complex type of detections that can viewed by the NER classifier as an intermittent set of detections.

    E.g.,
    TEXT:       Calle Serrano, 32. P5, planta 7. Puerta 2B. Madrid
    DETECTIONS:       AAAAAAAAAAAAAAAAA         AAAAAAAA AA AAAAAA
    (NOTE: 'A' stands for 'ADDRESS' in this context)

This step identifies these clear cases, where address detections are split holding undetected chunks in between, and merges them.

### Extract manual tokens from the text

This step uses traditional pattern matching to identify clear elements. Since the NLP-NER task provides the main functionality, the objective for this step is just to complement the NER detections.

The manually extracted detections will have a lower certainty than the NER ones. This allows for the system to always choose the NER one over the manual one if it doubts.

There are five main manual detections extracted:
- Email
- Phone
- Passwords
- Tokens with digits
- Names

Emails, phones, passwords and tokens-with-digits use regular expressions:

- Email:
  
      [A-Za-z0-9._-]+@[A-Za-z0-9.-]+\\.[A-Za-z0-9]+

- Phone::
  
      ((\\(?\\+?\\d{1,4}[\\)\\-]{0,2})?((\\d{3}([\\.\\-]?)(\\d{2}\\5){2}\\d{2})|(\\d{2}([\\.\\-]?)\\d{3}\\8\\d{2}\\8\\d{2})))

- Passwords:
  
      ([.\\S]?(((?=.*?\\S{6,})(?=([\\p{L}]+[0-9@!@#$%^&\\-*\\{\\}\\[\\]<>\\,\\.;:\"'~`\\|+\\-\\(\\)\\/\\?=\\\\_]+[0-9@!@#$%^&\\-*\\{\\}\\[\\]<>\\,\\.;:\"'~`\\|+\\-\\(\\)\\/\\?=\\\\_\\p{L}]+))([\\p{L}]+[0-9@!@#$%^&\\-*\\{\\}\\[\\]<>\\,\\.;:\"'~`\\|+\\-\\(\\)\\/\\?=\\\\_]+[0-9\\p{L}@!@#$%^&\\-*\\{\\}\\[\\]<>\\,\\.;:\"'~`\\|+\\-\\(\\)\\/\\?=\\\\_ñÑ]*))|([0-9]{4})|((?=.*?\\S{6,})([\\p{L}]{1}[\\p{Ll}]*[\\p{Lu}]+[\\p{L}0-9]*.*))))

- Tokens with digits:

      ((?<=\\s|^|,)((([a-z-\\/\\.,A-Z]*[0-9]+[a-z-\\/\\.,A-Z]**)(\\4)*))+(?=,?\\s|$))

**NOTE:** the regular expressions listed above use java escaping format.

The names are guessed using a set of dictionaries. These dictionaries are compiled from the most common names in Spain and in the world. There are three dictionaries:
- first_names.txt
- last_names.txt
- name_complements.txt

These dictionaries are embedded within the artifacts and seamlessly used.

### Translate detections to semantic terms.

The protection system holds two main translations towards the objective of guessing the best security label of every detection:

NLP label (also manual ones) ---------> semantic label

Semantic label ----------------------------> security label

On this step, the NLP labels are transformed into semantic labels. The translation is performed using the mapping between them. The mapping is directed by a json file included in the model directory structure. This json can be accessed and maintained. It has the name:

**NlpToSemanticTerm.json**

**Note:** the full json can be accessed in this link:

https://drive.google.com/file/d/1W0NClpfXvA0sW4tokPl8DMSzSNlhQ6IC/view?usp=sharing

### Link detections to data properties using natural language terms.

Using the semantic label assigned to a detection, in this step the system will gather all of the possible combinations of data properties (RCN fields) that are described as that semantic label.

There is a prior assignment of 'semantic label' to 'RCN field' loaded from the patterns file (see integration document). Since this definition can be redundant -one semantic label can describe several fields- all of the possible combinations are linked to each de detections.

This information will later allow the system to decide the best security label per detection.

### Clean low information detections for names

This step is a help for the name detector. It aims to delete detections tagged as 'NAME' that hold a low information level. In order to do this, the system reviews every 'NAME' detection and checks if the information level is too low.

This step helps reduce overprotection.

An example of its effects:

    Full text:        Consuelo es lo que no tengo cuando llamo al Call Center.
    Orig. detection:  NNNNNNNN ...............................................
    Output detection: ........................................................

**NOTE:** 'N' represents detection type 'NAME'

### Add detections using data patterns.

The system allows defining types of entities by describing their data patterns. This capability makes use of the pattern definitions held by the server for the quality rules management. Even though they can be manually entered in the server, as quality rules for the ontology “data properties”, the initial setup allows loading them in the patterns file (see integration document).

These patterns allow the system to “double check” if a detection should be something else. This means that, if the NLP gets the wrong type for a detection, this step will try to recover the best match between detection and type.

An example of this would be the following: a new security class ‘B’ is introduced that is typically detected by the NLP module as ‘A’. If the data pattern is definable as a regular expression (other options are also available), we can instruct the system to override the NLP output and set the right label. Please note that the system did not need new training. Even if the NLP model is re-trained, it is still helped by this feature.

The result of this step is that the detections are populated with possible data properties they can represent.

### Select the most restrictive security label per detection.

In this step, the system reviews all the possible RCN fields per detection -as stored in its data properties- and chooses the one that is most restrictive.

E.g., if the system doubts between the options ‘ACCOUNT_NUMBER’ and ‘DEVICE’ for a detection, the most secure one will be chosen. In this case ‘ACCOUNT_NUMBER’.

This selection is driven by the metadata assigned as an attribute to each data property. These attributes come from the patterns file (see integration document) and will be shown in the server as the attribute ‘rcn.protection_prevalence’.

The lowest prevalence data property will be selected in order to define the security label.

**Note:** if two different alternatives are available both with the lowest value (e.g., 01) the selection is arbitrary.

### Deduplicate detections.
This step just makes sure there are not more than one detection per block of text. If there are more than one, they are merged and the most restrictive method one is chosen. This could happen due to an overlap of the detections when using different protection techniques.

Example:

    Text:               Mi telefono y mi pin son 95 234 54 32 1234
    Detection 1:        .........................PPPPPPPPPPPPP....
    Detection 2:        ...................................KKKKKKK
    Output detections:  .........................KKKKKKKKKKKKKKKKK
    
Notes for the example:
- ‘P’ holds for ‘PHONE’ detection type.
- ‘K’ holds for ‘PIN_KEY’ detection type.
- ‘PIN_KEY’ has a higher prevalence (00) than ‘PHONE’ (03).
    
### Ensure security label.

This step is a prevention method. In no case there should be a detection without a security label. But if this happens in the future (code changes, data gets broken, config issues, ...) the system will at least assign the semantic label as the security label.

The semantic labels were designed keeping this in mind and their names match with the initial set of security labels.

### Tag gaps between consecutive detections.

This mechanism aims to find chunks of text that are not part of a detection but could be merged into one. This step is the consequence of several fine-tuning iterations on the address, name and manual token detection abilities.

NLP modules typically process texts performing several tokenization passes. The first one tries to separate the sentences. The most common way this is done is by splitting by the dot between them. Since a detection could contain dots (e.g., the address: “Serrano, 22. P 2b. 28456, Madrid”) the NLP tokenizer could end up splitting the detection in two sentences, hence creating two detections usually separated by some characters.

This step analyzes gaps between consecutive detections and creates ad hoc detections for them if needed.

The connecting detection will be used in a later step to complete the detection.

### Remove unwanted detections.

The system knows about more labels than the ones it wants to finally report. An example is the “TRANSACTION_DATE” label. This extra label exists to improve the system precision: if it knows about other types of dates, it is able to better find the birth dates and reduce the false positives -reducing overprotection-.

When returning the results, the extra labels need to be avoided. This method just filters them out in order to not report on them.

    Text:               Naci el 3-4-86. El dinero lo enviaron el 8-5-87
    Raw detections:     ........BBBBBB...........................TTTTTT
    Output detections:  ........BBBBBB.................................
    
Notes for the example:
- ‘B’ means the character belongs to a ‘BIRTH_DATE’ detection.
- ‘T’ means the character belongs to a ‘TRANSACTION_DATE’ detection.
- ‘TRANSACTION_DATE’ is not supposed to be reported.

### Merge consecutive detections with the same security label.

This step is in charge of the last merging of consecutive detections. After the system possibly created some connecting detections -see step 3.13- this step identifies them and merges them back into one single one.

**Note:** This step follows the requirements of several fine-tuning iterations on the final system
    