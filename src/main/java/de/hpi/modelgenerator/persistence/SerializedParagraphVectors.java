package de.hpi.modelgenerator.persistence;


import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.springframework.data.annotation.Id;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;

@Setter
@Getter
@JsonIgnoreProperties(ignoreUnknown = true)
@NoArgsConstructor
public class SerializedParagraphVectors {

    @Id private String networkType;
    private byte[] serializedNeuralNetwork;

    public SerializedParagraphVectors(ParagraphVectors vectors) {
        OutputStream out = new ByteArrayOutputStream(100000000);
        try {
            WordVectorSerializer.writeParagraphVectors(vectors, out);
        } catch (IOException e) {
            e.printStackTrace();
        }

        byte[] serializedNetwork = ((ByteArrayOutputStream) out).toByteArray();
        setSerializedNeuralNetwork(serializedNetwork);
    }
}
