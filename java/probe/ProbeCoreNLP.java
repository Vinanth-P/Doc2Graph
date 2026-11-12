package probe;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;

import java.util.List;
import java.util.Properties;

public class ProbeCoreNLP {
    public static void main(String[] args) {
        System.err.println("Initializing Stanford CoreNLP probe...");
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,relation");
        props.setProperty("ner.model", "edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz");
        props.setProperty("ner.useSUTime", "false");
        props.setProperty("ner.applyNumericClassifiers", "false");
        props.setProperty("ner.applyFineGrained", "false");
        props.setProperty("parse.model", "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz");
        props.setProperty("pos.model", "edu/stanford/nlp/models/pos-tagger/english-left3words-distsim.tagger");
        props.setProperty("sup.relation.model", "edu/stanford/nlp/models/supervised_relation_extractor/roth_relation_model_pipelineNER.ser");

        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        System.err.println("Pipeline initialized successfully!");

        String text = "Barack Obama was born in Hawaii. Bill Gates founded Microsoft in Albuquerque, New Mexico. Satya Nadella works for Microsoft in Redmond.";
        Annotation doc = new Annotation(text);
        pipeline.annotate(doc);

        List<CoreMap> sentences = doc.get(CoreAnnotations.SentencesAnnotation.class);
        System.err.println("Sentences count: " + sentences.size());

        for (int i = 0; i < sentences.size(); i++) {
            CoreMap sentence = sentences.get(i);
            System.out.println("--- Sentence " + (i + 1) + ": " + sentence.get(CoreAnnotations.TextAnnotation.class));

            List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
            for (CoreLabel token : tokens) {
                String ner = token.get(CoreAnnotations.NamedEntityTagAnnotation.class);
                if (ner != null && !ner.equals("O")) {
                    System.out.println("  NER Token: " + token.word() + " [" + ner + "] (" + token.beginPosition() + "-" + token.endPosition() + ")");
                }
            }

            List<RelationMention> rels = sentence.get(MachineReadingAnnotations.RelationMentionsAnnotation.class);
            if (rels == null || rels.isEmpty()) {
                System.out.println("  No relation mentions found.");
            } else {
                for (RelationMention rm : rels) {
                    Counter<String> probs = rm.getTypeProbabilities();
                    System.out.println("  Relation: type=" + rm.getType() + ", probs=" + (probs != null ? probs.toString() : "null"));
                    List<EntityMention> entArgs = rm.getEntityMentionArgs();
                    if (entArgs != null) {
                        for (int a = 0; a < entArgs.size(); a++) {
                            EntityMention em = entArgs.get(a);
                            int cStart = tokens.get(em.getExtentTokenStart()).beginPosition();
                            int cEnd = tokens.get(em.getExtentTokenEnd() - 1).endPosition();
                            int hStart = tokens.get(em.getHeadTokenStart()).beginPosition();
                            int hEnd = tokens.get(em.getHeadTokenEnd() - 1).endPosition();
                            System.out.println("    Arg " + a + ": type=" + em.getType() + ", text=" + em.getValue()
                                    + ", extent=[" + cStart + "-" + cEnd + "]"
                                    + ", head=[" + hStart + "-" + hEnd + "]");
                        }
                    }
                }
            }
        }
    }
}
