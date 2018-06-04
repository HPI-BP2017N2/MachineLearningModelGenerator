package de.hpi.modelgenerator.api;

import de.hpi.modelgenerator.persistence.ClassifierTrainingState;
import de.hpi.modelgenerator.services.ModelGeneratorService;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiResponse;
import io.swagger.annotations.ApiResponses;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

@RestController
@Slf4j
@Getter(AccessLevel.PRIVATE)
@RequiredArgsConstructor
public class ModelGeneratorController {

    private final ModelGeneratorService service;
    private final ClassifierTrainingState categoryClassifierTrainingState = new ClassifierTrainingState();
    private final ClassifierTrainingState brandClassifierTrainingState = new ClassifierTrainingState();
    private final ClassifierTrainingState modelTrainingState = new ClassifierTrainingState();

    @ApiOperation(value = "Generate category classifier")
    @ApiResponses(value = {
            @ApiResponse(code = 200, message = "Successfully generated category classifier."),
            @ApiResponse(code = 500, message = "There was an error (serializing the classifier).")})
    @RequestMapping(value = "/generateCategoryClassifier", method = RequestMethod.POST)
    public void generateCategoryClassifier() throws IOException {
        getService().generateCategoryClassifier(getCategoryClassifierTrainingState());

    }

    @ApiOperation(value = "Generate brand classifier")
    @ApiResponses(value = {
            @ApiResponse(code = 200, message = "Successfully generated brand classifier."),
            @ApiResponse(code = 500, message = "There was an error (serializing the classifier).")})
    @RequestMapping(value = "/generateBrandClassifier", method = RequestMethod.POST)
    public void generateBrandClassifier() throws IOException {
        getService().generateBrandClassifier(getBrandClassifierTrainingState());

    }

    @ApiOperation(value = "Generate model")
    @ApiResponses(value = {
            @ApiResponse(code = 200, message = "Successfully generated model."),
            @ApiResponse(code = 500, message = "There was an error (serializing the model). Make sure that brand classifier is present.")})
    @RequestMapping(value = "/generateModel", method = RequestMethod.POST)
    public void generateModel() throws IOException {
        getService().generateModel(getModelTrainingState());
    }

    @ApiOperation(value = "Generate all classifiers")
    @ApiResponses(value = {
            @ApiResponse(code = 200, message = "Successfully generated all classifier."),
            @ApiResponse(code = 500, message = "There was an error (serializing the classifiers).")})
    @RequestMapping(value = "/generateAllClassifiers", method = RequestMethod.POST)
    public void generateAllClassifiers() throws IOException {
        getService().refreshTrainingAndTestingSet();
        generateCategoryClassifier();
        generateBrandClassifier();
        generateModel();
        getService().freeTestingSet();
    }
}
