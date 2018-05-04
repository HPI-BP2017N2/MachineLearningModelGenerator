package de.hpi.modelgenerator.api;

import de.hpi.modelgenerator.services.ModelGeneratorService;
import de.hpi.modelgenerator.services.ParagraphVectorsClassifierExample;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import java.io.FileNotFoundException;

@RestController
@Slf4j
@Getter(AccessLevel.PRIVATE)
@RequiredArgsConstructor
public class ModelGeneratorController {

    private final ModelGeneratorService service;
    private final ParagraphVectorsClassifierExample neuralTrainer;

    @RequestMapping(value = "/train/{shopId}", method = RequestMethod.GET, produces = "application/json")
    public void doSth(@PathVariable long shopId){
        try {
            getNeuralTrainer().makeParagraphVectors(shopId);
        } catch (Exception e) {
            e.printStackTrace();
        }

        try {
            getNeuralTrainer().checkUnlabeledData();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


}
