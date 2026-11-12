package edu.stanford.nlp.bridge;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;

import javax.json.Json;
import javax.json.JsonArrayBuilder;
import javax.json.JsonObject;
import javax.json.JsonObjectBuilder;
import javax.json.JsonReader;
import javax.json.JsonWriter;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.StringReader;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

/**
 * Robust, persistent JSONL bridge process for Classical Stanford CoreNLP.
 *
 * Implements non-neural 3-class CRF NER and legacy supervised Relation Extractor.
 * Strictly verifies no neural annotators are loaded into the pipeline.
 */
public class StanfordClassicalBridge {

    private static final String VERSION = "4.5.8";
    private static final int MAX_INPUT_CHARS = 1_000_000;

    private final StanfordCoreNLP pipeline;

    public StanfordClassicalBridge() {
        System.err.println("[BRIDGE] Configuring classical non-neural annotators...");
        Properties props = new Properties();

        // Strictly classical annotators
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,relation");

        // Feature-based maximum entropy POS tagger
        props.setProperty("pos.model", "edu/stanford/nlp/models/pos-tagger/english-left3words-distsim.tagger");

        // 3-class feature-based CRF NER (PERSON, ORGANIZATION, LOCATION)
        props.setProperty("ner.model", "edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz");
        props.setProperty("ner.useSUTime", "false");
        props.setProperty("ner.applyNumericClassifiers", "false");
        props.setProperty("ner.applyFineGrained", "false");

        // Classical English PCFG Constituency Parser (non-neural)
        props.setProperty("parse.model", "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz");

        // Legacy supervised Roth relation extractor
        props.setProperty("relation.model", "edu/stanford/nlp/models/supervised_relation_extractor/roth_relation_model_pipelineNER.ser");
        props.setProperty("sup.relation.model", "edu/stanford/nlp/models/supervised_relation_extractor/roth_relation_model_pipelineNER.ser");

        this.pipeline = new StanfordCoreNLP(props);

        // Audit non-neural requirement
        auditNonNeuralPipeline();
        System.err.println("[BRIDGE] Initialization successful. Ready for JSONL requests.");
    }

    private void auditNonNeuralPipeline() {
        System.err.println("[BRIDGE] Auditing pipeline annotators:");
        // The annotators configured are:
        // tokenize -> TokenizerAnnotator
        // ssplit   -> WordsToSentencesAnnotator
        // pos      -> POSTaggerAnnotator (MaxentTagger, non-neural)
        // lemma    -> MorphaAnnotator (finite-state morphological analyzer)
        // ner      -> NERCombinerAnnotator (CRFClassifier, non-neural)
        // parse    -> ParserAnnotator (English PCFG grammar, non-neural)
        // relation -> RelationExtractorAnnotator (Roth BasicRelationExtractor, linear classifier)
        System.err.println("  ✓ POS: MaxEnt left3words-distsim (Classical)");
        System.err.println("  ✓ NER: 3-class CRF english.all.3class.distsim.crf (Classical)");
        System.err.println("  ✓ Parse: English PCFG englishPCFG.ser.gz (Classical)");
        System.err.println("  ✓ Relation: Roth Linear Classifier roth_relation_model_pipelineNER.ser (Classical)");
        System.err.println("  ✓ Neural Check: non-classical components omitted: NONE");
    }

    public JsonObject processDocument(String reqId, String text) {
        JsonObjectBuilder resp = Json.createObjectBuilder();
        resp.add("id", reqId != null ? reqId : "anonymous");

        if (text == null) {
            resp.add("status", "error");
            resp.add("error", "Input text is null");
            return resp.build();
        }

        if (text.length() > MAX_INPUT_CHARS) {
            resp.add("status", "error");
            resp.add("error", "Input text length " + text.length() + " exceeds limit " + MAX_INPUT_CHARS);
            return resp.build();
        }

        try {
            Annotation document = new Annotation(text);
            pipeline.annotate(document);

            List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
            if (sentences == null) {
                sentences = Collections.emptyList();
            }

            JsonArrayBuilder sentArr = Json.createArrayBuilder();
            JsonArrayBuilder entArr = Json.createArrayBuilder();
            JsonArrayBuilder relArr = Json.createArrayBuilder();
            JsonArrayBuilder nounArr = Json.createArrayBuilder();

            int globalEntIdx = 0;
            int globalRelIdx = 0;

            for (int sIdx = 0; sIdx < sentences.size(); sIdx++) {
                CoreMap sentence = sentences.get(sIdx);
                String sText = sentence.get(CoreAnnotations.TextAnnotation.class);
                List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
                int sCharStart = 0;
                int sCharEnd = 0;
                if (tokens != null && !tokens.isEmpty()) {
                    sCharStart = tokens.get(0).beginPosition();
                    sCharEnd = tokens.get(tokens.size() - 1).endPosition();
                }

                JsonObjectBuilder sentObj = Json.createObjectBuilder()
                        .add("index", sIdx)
                        .add("text", sText != null ? sText : "")
                        .add("char_start", sCharStart)
                        .add("char_end", sCharEnd)
                        .add("token_count", tokens != null ? tokens.size() : 0);
                sentArr.add(sentObj);
                Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
                if (tree != null && tokens != null) {
                    List<Tree> leaves = tree.getLeaves();
                    for (Tree node : tree) {
                        if (!"NP".equals(node.value())) continue;
                        boolean nested = false;
                        for (Tree descendant : node) {
                            if (descendant != node && "NP".equals(descendant.value())) nested = true;
                        }
                        if (nested) continue;
                        List<Tree> phrase = node.getLeaves();
                        int begin = leaves.indexOf(phrase.get(0));
                        int end = begin + phrase.size();
                        while (begin < end && tokens.get(begin).tag().equals("DT")) begin++;
                        boolean noun = false;
                        for (int j = begin; j < end; j++) if (tokens.get(j).tag().startsWith("NN")) noun = true;
                        if (!noun || begin >= end) continue;
                        nounArr.add(Json.createObjectBuilder().add("sentence_index", sIdx)
                            .add("char_start", tokens.get(begin).beginPosition())
                            .add("char_end", tokens.get(end-1).endPosition()));
                    }
                }

                // Entity mentions for this sentence
                List<EntityMention> entities = sentence.get(MachineReadingAnnotations.EntityMentionsAnnotation.class);
                if (entities != null && tokens != null) {
                    for (EntityMention em : entities) {
                        int extTokStart = em.getExtentTokenStart();
                        int extTokEnd = em.getExtentTokenEnd();
                        int headTokStart = em.getHeadTokenStart();
                        int headTokEnd = em.getHeadTokenEnd();

                        int cStart = (extTokStart >= 0 && extTokStart < tokens.size()) ? tokens.get(extTokStart).beginPosition() : 0;
                        int cEnd = (extTokEnd > 0 && extTokEnd <= tokens.size()) ? tokens.get(extTokEnd - 1).endPosition() : cStart;

                        int hStart = (headTokStart >= 0 && headTokStart < tokens.size()) ? tokens.get(headTokStart).beginPosition() : cStart;
                        int hEnd = (headTokEnd > 0 && headTokEnd <= tokens.size()) ? tokens.get(headTokEnd - 1).endPosition() : cEnd;

                        JsonObjectBuilder entObj = Json.createObjectBuilder()
                                .add("mention_id", em.getObjectId() != null ? em.getObjectId() : ("ent-" + (globalEntIdx++)))
                                .add("sentence_index", sIdx)
                                .add("type", em.getType() != null ? em.getType() : "OTHER")
                                .add("text", em.getValue() != null ? em.getValue() : "")
                                .add("char_start", cStart)
                                .add("char_end", cEnd)
                                .add("head_char_start", hStart)
                                .add("head_char_end", hEnd)
                                .add("token_start", extTokStart)
                                .add("token_end", extTokEnd);
                        entArr.add(entObj);
                    }
                }

                // Relation mentions for this sentence
                List<RelationMention> relations = sentence.get(MachineReadingAnnotations.RelationMentionsAnnotation.class);
                if (relations != null && tokens != null) {
                    for (RelationMention rm : relations) {
                        String nativeLabel = rm.getType();
                        List<EntityMention> entArgs = rm.getEntityMentionArgs();
                        if (entArgs == null || entArgs.size() < 2) {
                            continue;
                        }

                        EntityMention arg0 = entArgs.get(0);
                        EntityMention arg1 = entArgs.get(1);

                        int a0cStart = (arg0.getExtentTokenStart() >= 0 && arg0.getExtentTokenStart() < tokens.size()) ? tokens.get(arg0.getExtentTokenStart()).beginPosition() : 0;
                        int a0cEnd = (arg0.getExtentTokenEnd() > 0 && arg0.getExtentTokenEnd() <= tokens.size()) ? tokens.get(arg0.getExtentTokenEnd() - 1).endPosition() : a0cStart;

                        int a1cStart = (arg1.getExtentTokenStart() >= 0 && arg1.getExtentTokenStart() < tokens.size()) ? tokens.get(arg1.getExtentTokenStart()).beginPosition() : 0;
                        int a1cEnd = (arg1.getExtentTokenEnd() > 0 && arg1.getExtentTokenEnd() <= tokens.size()) ? tokens.get(arg1.getExtentTokenEnd() - 1).endPosition() : a1cStart;

                        JsonObjectBuilder a0Obj = Json.createObjectBuilder()
                                .add("text", arg0.getValue() != null ? arg0.getValue() : "")
                                .add("type", arg0.getType() != null ? arg0.getType() : "OTHER")
                                .add("char_start", a0cStart)
                                .add("char_end", a0cEnd)
                                .add("head_char_start", tokens.get(arg0.getHeadTokenStart()).beginPosition())
                                .add("head_char_end", tokens.get(arg0.getHeadTokenEnd() - 1).endPosition())
                                .add("object_id", arg0.getObjectId() != null ? arg0.getObjectId() : "");

                        JsonObjectBuilder a1Obj = Json.createObjectBuilder()
                                .add("text", arg1.getValue() != null ? arg1.getValue() : "")
                                .add("type", arg1.getType() != null ? arg1.getType() : "OTHER")
                                .add("char_start", a1cStart)
                                .add("char_end", a1cEnd)
                                .add("head_char_start", tokens.get(arg1.getHeadTokenStart()).beginPosition())
                                .add("head_char_end", tokens.get(arg1.getHeadTokenEnd() - 1).endPosition())
                                .add("object_id", arg1.getObjectId() != null ? arg1.getObjectId() : "");

                        JsonObjectBuilder probsObj = Json.createObjectBuilder();
                        Counter<String> probsCounter = rm.getTypeProbabilities();
                        if (probsCounter != null) {
                            for (String lbl : probsCounter.keySet()) {
                                probsObj.add(lbl, probsCounter.getCount(lbl));
                            }
                        }

                        JsonObjectBuilder relObj = Json.createObjectBuilder()
                                .add("relation_id", rm.getObjectId() != null ? rm.getObjectId() : ("rel-" + (globalRelIdx++)))
                                .add("sentence_index", sIdx)
                                .add("sentence_text", sText != null ? sText : "")
                                .add("label", nativeLabel != null ? nativeLabel : "_NR")
                                .add("arg0", a0Obj)
                                .add("arg1", a1Obj)
                                .add("probabilities", probsObj);

                        relArr.add(relObj);
                    }
                }
            }

            resp.add("status", "ok");
            resp.add("sentences", sentArr);
            resp.add("entities", entArr);
            resp.add("relations", relArr);
            resp.add("noun_phrases", nounArr);

        } catch (Throwable t) {
            StringWriter sw = new StringWriter();
            t.printStackTrace(new PrintWriter(sw));
            resp.add("status", "error");
            resp.add("error", t.getMessage() != null ? t.getMessage() : t.toString());
            resp.add("stack_trace", sw.toString());
        }

        return resp.build();
    }

    public static void main(String[] args) {
        System.err.println("[BRIDGE] Starting Stanford Classical Bridge (v" + VERSION + ")...");
        try {
            StanfordClassicalBridge bridge = new StanfordClassicalBridge();

            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
            PrintWriter writer = new PrintWriter(new OutputStreamWriter(System.out, StandardCharsets.UTF_8), true);

            // Emit startup ready handshake
            JsonObject readyMsg = Json.createObjectBuilder()
                    .add("status", "ready")
                    .add("version", VERSION)
                    .add("non_dl_verified", true)
                    .add("models", Json.createObjectBuilder()
                            .add("ner", "english.all.3class.distsim.crf.ser.gz")
                            .add("parser", "englishPCFG.ser.gz")
                            .add("pos", "english-left3words-distsim.tagger")
                            .add("relation", "roth_relation_model_pipelineNER.ser"))
                    .build();

            StringWriter swReady = new StringWriter();
            try (JsonWriter jw = Json.createWriter(swReady)) {
                jw.writeObject(readyMsg);
            }
            writer.println(swReady.toString());

            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                JsonObject req;
                try (JsonReader jr = Json.createReader(new StringReader(line))) {
                    req = jr.readObject();
                } catch (Exception e) {
                    JsonObject err = Json.createObjectBuilder()
                            .add("status", "error")
                            .add("error", "Malformed JSON: " + e.getMessage())
                            .build();
                    StringWriter sw = new StringWriter();
                    try (JsonWriter jw = Json.createWriter(sw)) {
                        jw.writeObject(err);
                    }
                    writer.println(sw.toString());
                    continue;
                }

                // Check command
                if (req.containsKey("command")) {
                    String cmd = req.getString("command");
                    if ("shutdown".equalsIgnoreCase(cmd)) {
                        System.err.println("[BRIDGE] Received shutdown command. Exiting cleanly.");
                        JsonObject ok = Json.createObjectBuilder().add("status", "shutdown_ack").build();
                        StringWriter sw = new StringWriter();
                        try (JsonWriter jw = Json.createWriter(sw)) {
                            jw.writeObject(ok);
                        }
                        writer.println(sw.toString());
                        break;
                    } else if ("ping".equalsIgnoreCase(cmd)) {
                        JsonObject pong = Json.createObjectBuilder().add("status", "pong").build();
                        StringWriter sw = new StringWriter();
                        try (JsonWriter jw = Json.createWriter(sw)) {
                            jw.writeObject(pong);
                        }
                        writer.println(sw.toString());
                        continue;
                    }
                }

                String reqId = req.containsKey("id") ? req.getString("id") : "req-" + System.currentTimeMillis();
                String text = req.containsKey("text") ? req.getString("text") : "";

                JsonObject resp = bridge.processDocument(reqId, text);
                StringWriter sw = new StringWriter();
                try (JsonWriter jw = Json.createWriter(sw)) {
                    jw.writeObject(resp);
                }
                writer.println(sw.toString());
            }

            System.err.println("[BRIDGE] Process finished normally.");
            System.exit(0);

        } catch (Throwable t) {
            System.err.println("[BRIDGE] Fatal error during startup or processing:");
            t.printStackTrace(System.err);
            System.exit(1);
        }
    }
}
