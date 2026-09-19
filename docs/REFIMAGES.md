Implementa in `o1.c` un sistema di **reference aliases / reference tags** che permetta di associare un nome semantico a ogni `--ref-image` e di usare quel nome nel prompt.

L'obiettivo è ottenere una CLI del tipo:

```bash
./build/hidream \
  --model dev \
  --ref-image person=alice.png \
  --ref-image outfit=jacket.png \
  --ref-image pose=pose.png \
  --prompt '@person wearing @outfit, using the body pose from @pose'
```

IMPORTANTE: questa feature deve essere implementata principalmente come **frontend/prompt expansion**.

NON modificare:

* vision tower math
* image embeddings
* DeepStack math
* reference ordering
* scheduler
* tokenizer vocabulary
* embedding matrix
* CUDA kernels

Non aggiungere nuovi special token al modello.

`@person`, `@outfit`, ecc. NON sono token appresi dal modello: sono alias interpretati dal runtime di `o1.c`.

---

## 1. Semantica fondamentale

HiDream riceve le reference in un ordine deterministico:

```text
ref[0]
ref[1]
ref[2]
...
```

e quell'ordine deve continuare a essere IDENTICO per entrambi i conditioning path:

```text
PATH A
ref image
-> full-resolution ref patches
-> vinputs
```

e:

```text
PATH B
ref image
-> VLM preprocessing
-> vision tower
-> image_embeds/deepstack
```

Gli alias sono solamente metadata associati alle reference.

Esempio:

```text
ref[0] alias = person
ref[1] alias = outfit
ref[2] alias = pose
```

Il modello continua a ricevere:

```text
image 1 = ref[0]
image 2 = ref[1]
image 3 = ref[2]
```

Non deve avvenire alcun reorder in base all'uso degli alias nel prompt.

---

## 2. Sintassi CLI

Mantieni compatibilità con la sintassi esistente:

```bash
--ref-image photo.png
```

e aggiungi:

```bash
--ref-image person=photo.png
```

Entrambe devono funzionare.

Per reference senza nome assegna automaticamente:

```text
@ref1
@ref2
@ref3
...
```

Esempio:

```bash
--ref-image a.png
--ref-image shirt=b.png
```

produce concettualmente:

```text
ref[0]:
    automatic_alias = ref1
    explicit_alias  = none

ref[1]:
    automatic_alias = ref2
    explicit_alias  = shirt
```

Quindi possono essere usati:

```text
@ref1
@shirt
```

Gli alias automatici `@refN` devono esistere SEMPRE, anche quando è presente un alias esplicito.

Esempio:

```bash
--ref-image alice=alice.png
```

può essere riferita sia come:

```text
@alice
```

sia come:

```text
@ref1
```

---

## 3. Parsing robusto

Non interpretare ogni `=` nel path in modo ingenuo.

Implementa un parser centralizzato per la ref specification, ad esempio:

```c
typedef struct {
    char *path;
    char *alias;          /* explicit alias, optional */
    int user_index;
    int effective_index;
    int is_internal;
} hd_reference;
```

oppure estendi la struttura reference esistente senza duplicare il modello dati.

Regole alias:

```text
[A-Za-z_][A-Za-z0-9_-]*
```

Consenti quindi:

```text
person
person_1
dress-blue
pose_front
```

Rifiuta alias:

* vuoti
* duplicati
* contenenti whitespace
* che iniziano con `__`
* uguali a reserved internal aliases

Errore esplicito:

```text
duplicate reference alias: person
```

Non accettare silenziosamente ambiguità.

---

## 4. Internal references vs user references

Alcune modalità di `o1.c` possono creare reference aggiuntive internamente:

```text
layout canvas
skeleton
pose/control references
future generated refs
```

Queste NON devono rompere gli alias assegnati dall'utente.

Separare chiaramente:

```text
USER REFERENCES
person
shirt
pose

INTERNAL REFERENCES
__layout
__skeleton1
__skeleton2
```

Gli alias che iniziano con:

```text
__
```

sono riservati al runtime.

Per esempio:

```text
user ref 0 -> person
user ref 1 -> dress
internal ref -> __layout
```

effective list:

```text
effective_ref[0] = person
effective_ref[1] = dress
effective_ref[2] = __layout
```

L'aggiunta di `__layout` non deve cambiare il significato di:

```text
@person
@dress
```

---

## 5. Alias table

Dopo che tutte le effective refs sono state costruite, crea una tabella deterministica:

```text
alias           effective_ref_index
-----------------------------------
ref1            0
person          0

ref2            1
dress           1

ref3            2
__layout        2
```

Questa tabella deve essere usata SOLO per costruire il prompt expanded.

Non usarla per reorder dei tensor.

---

## 6. Prompt expansion

Prima della tokenizzazione, espandi il prompt originale.

Esempio input:

```text
@person wearing @dress while standing in the pose shown by @pose
```

con:

```text
person -> ref 1
dress  -> ref 2
pose   -> ref 3
```

deve diventare semanticamente qualcosa come:

```text
Reference image 1 is named "person".
Reference image 2 is named "dress".
Reference image 3 is named "pose".

Use the subject shown in reference image 1 ("person") wearing the clothing
shown in reference image 2 ("dress"), while using the body pose shown in
reference image 3 ("pose").
```

NON limitarti alla sostituzione:

```text
@person -> reference image 1
```

Aggiungi anche un piccolo header che stabilisca chiaramente l'associazione nome→immagine.

Formato suggerito:

```text
Reference image mapping:
- "person" = reference image 1
- "dress" = reference image 2
- "pose" = reference image 3

User request:
Use the subject from reference image 1 ("person") wearing the clothing from
reference image 2 ("dress"), using the pose from reference image 3 ("pose").
```

Mantienilo breve: non vogliamo gonfiare inutilmente il prompt.

---

## 7. Espansione degli alias nel body

Fai tokenizzazione/scan lessicale, non semplici `strstr()` ripetuti che possono creare collisioni.

Esempio:

```text
@person
@person2
```

non devono interferire.

Sostituzione consigliata:

```text
@person
```

->

```text
reference image 1 ("person")
```

Quindi:

```text
A portrait of @person
```

diventa:

```text
A portrait of reference image 1 ("person")
```

Alias sconosciuto deve essere errore:

```text
unknown reference alias: @foo
```

NON lasciarlo nel prompt senza warning.

---

## 8. Prompt senza alias

Se il prompt NON contiene `@...`, il comportamento esistente deve rimanere esattamente invariato.

NON aggiungere automaticamente il reference mapping header se nessun alias viene utilizzato.

Questo è importante per preservare la compatibilità con prompt e fixture upstream.

Quindi:

```text
refs presenti + nessun @alias
```

=> comportamento attuale HiDream.

Solo:

```text
refs presenti + almeno un @alias
```

=> prompt expansion.

---

## 9. Alias automatici

Supporta:

```text
@ref1
@ref2
...
```

senza bisogno di specificare nomi espliciti.

Esempio:

```bash
--ref-image face.jpg \
--ref-image clothes.jpg \
--prompt 'Use the identity from @ref1 and clothes from @ref2'
```

Espansione:

```text
Reference image mapping:
- "ref1" = reference image 1
- "ref2" = reference image 2
...
```

Questo dà immediatamente una sintassi utile anche senza named refs.

---

## 10. Named refs

Esempio principale da supportare:

```bash
./build/hidream \
  --model dev \
  --ref-image woman=woman.png \
  --ref-image dress=dress.png \
  --ref-image pose=pose.png \
  --prompt '@woman wearing @dress using the body pose from @pose'
```

Expected mapping:

```text
woman -> image 1
dress -> image 2
pose -> image 3
```

e nessun cambiamento all'ordine dei tensor visuali.

---

## 11. Non introdurre veri special tokens

NON fare:

```text
<person>
<dress>
<pose>
```

come nuovi tokenizer tokens.

NON modificare:

```text
tokenizer vocabulary
embed_tokens
151936 vocab size
model weights
```

Non sono token addestrati e quindi non avrebbero una semantica utile.

Gli alias devono sparire prima della tokenizzazione.

---

## 12. Mantieni separata un'eventuale futura modalità "interleaved labels"

NON implementare ora una modalità che cambia il multimodal message da:

```text
[IMAGE]
[IMAGE]
[IMAGE]
[text]
```

a:

```text
[IMAGE]
"person"
[IMAGE]
"dress"
...
```

Questa soluzione sarebbe più invasiva perché cambia:

* token layout
* position IDs
* `<image_pad>` positions
* sequence fixtures
* training-distribution assumptions

Per ora mantenere esattamente il formato upstream e fare soltanto prompt alias expansion.

Documenta questa possibilità come FUTURE WORK, niente codice production ora.

---

## 13. API / C API

Non limitarla alla CLI.

Estendi l'API reference in modo che un client possa specificare:

```c
{
    .path = "alice.png",
    .alias = "person"
}
```

Se modificare l'ABI pubblica è invasivo, mantieni backward compatibility:

* vecchio campo/path continua a funzionare
* alias opzionale NULL

Non rompere programmi già compilabili contro l'API attuale se evitabile.

---

## 14. Debug / introspection

Aggiungi opzionalmente sotto verbose/debug:

```text
reference mapping:
  @ref1    -> ref[0] alice.png
  @person  -> ref[0] alice.png
  @ref2    -> ref[1] dress.png
  @dress   -> ref[1] dress.png
```

e:

```text
expanded prompt:
...
```

Non stampare automaticamente in produzione normale.

---

## 15. Sicurezza sui path

Parsing di:

```text
alias=path
```

non deve rompere path reali.

In particolare non assumere che un path contenente `=` sia necessariamente named-ref.

La parte prima di `=` deve essere trattata come alias SOLO se passa la validazione alias.

Altrimenti tutta la stringa rimane path.

Quindi:

```text
foo=bar.png
```

con `foo` alias valido -> named ref.

Un path ambiguo può essere disambiguato usando sintassi esplicita futura, ma per ora documenta la regola.

---

## 16. Test unitari

Aggiungi test per:

```text
--ref-image foo.png
-> @ref1

--ref-image person=foo.png
-> @person e @ref1

2 named refs
-> mapping stabile

duplicate alias
-> fail

unknown @alias in prompt
-> fail

reserved __alias
-> fail

prompt senza alias
-> byte-identical/non-expanded rispetto al comportamento precedente

@ref1 accanto a @ref10
-> no collision

layout internal ref aggiunta
-> alias utente invariati
```

Test importante:

```text
ref order prima expansion
==
ref order dopo expansion
```

nessun tensor ordering deve cambiare.

---

## 17. Test end-to-end

Una volta che il ref-image path corrente è stabile, esegui smoke:

### Single ref

```bash
--ref-image person=example_assets/edit/test.jpg \
--prompt 'Place @person in a snowy mountain landscape'
```

### Two refs

```bash
--ref-image person=person.jpg \
--ref-image clothing=clothing.jpg \
--prompt '@person wearing the clothing shown in @clothing'
```

### Three refs

```bash
--ref-image person=person.jpg \
--ref-image clothing=clothing.jpg \
--ref-image pose=pose.jpg \
--prompt '@person wearing @clothing using the pose from @pose'
```

Verifica almeno:

* nessun crash
* expanded prompt corretto
* ref order invariato
* visual token counts invariati
* output generation valida

Non richiedere che il modello obbedisca perfettamente semanticamente per dichiarare corretta l'implementazione software: la qualità di binding semantico è una proprietà del modello e va testata separatamente.

---

## 18. README

Aggiungi una sezione tipo:

```text
Named reference images
```

spiegando:

```bash
--ref-image person=person.jpg
--ref-image shirt=shirt.jpg
--prompt '@person wearing @shirt'
```

Spiega chiaramente:

```text
@name is a prompt-side alias.
It does not add a new model token or modify the reference image encoding.
References retain their original command-line order.
```

Documenta anche:

```text
@ref1
@ref2
...
```

come alias automatici.

---

## 19. Architettura desiderata

Alla fine il flusso deve essere:

```text
CLI / C API
      |
      v
reference specs
[path + optional alias]
      |
      v
effective reference list
(user refs + internal refs)
      |
      +-----------------------+
      |                       |
      v                       v
alias table               image pipeline
      |                       |
      v                       +--> PATH A ref_patches
prompt expansion              |
      |                       +--> PATH B vision tower
      v
tokenizer
      |
      v
existing hd_seq_build / hd_forward
```

L'alias table NON deve entrare nel CUDA/model layer.

È metadata frontend.

---

## 20. Possibile estensione futura

Documenta, ma NON implementare ora:

```text
--ref-image person=...
--ref-role person:identity
--ref-role dress:clothing
--ref-role pose:pose
--ref-role style:style
```

Potremmo in futuro avere expansion specifiche:

```text
identity:
"Use the identity of the subject shown in reference image N"

clothing:
"Use the clothing shown in reference image N"

pose:
"Use the body pose shown in reference image N"

style:
"Use the visual style shown in reference image N"
```

Per ora lascia che il significato venga espresso dal prompt dell'utente.

---

## 21. Non mischiare questo task con il current ref-image correctness task

Se il reference generation path attuale NON è ancora completamente validato, completa prima:

* step0 parity
* valid edit generation
* regressione T2I

Poi implementa aliasing.

Questa feature non deve essere usata per nascondere o correggere problemi del visual wiring.

---

## 22. Definition of done

Il task è chiuso quando:

```text
old --ref-image PATH continua a funzionare
--ref-image NAME=PATH funziona
@NAME expansion funziona
@refN funziona sempre
duplicate/unknown/reserved alias fail closed
prompt senza alias resta invariato
internal refs non cambiano gli alias utente
reference tensor order resta invariato
C API supporta alias opzionale
unit tests PASS
single/multi-ref smoke PASS
README aggiornato
no debug temporaneo
clean build
git status clean
```

Commit suggerito:

```text
feat(ref): add named reference aliases in prompts
```

Mantieni la feature piccola e frontend-only. Non modificare vision tower, decoder, scheduler o CUDA per implementare gli alias.
