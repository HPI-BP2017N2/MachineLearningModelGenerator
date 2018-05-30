package de.hpi.modelgenerator.api;

import de.hpi.modelgenerator.persistence.ClassifierTrainingState;
import de.hpi.modelgenerator.persistence.repo.ModelRepository;
import de.hpi.modelgenerator.services.ModelGeneratorService;
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
    private final ModelRepository modelRepository;
    private final ClassifierTrainingState categoryClassifierTrainingState = new ClassifierTrainingState();
    private final ClassifierTrainingState brandClassifierTrainingState = new ClassifierTrainingState();
    private final ClassifierTrainingState modelTrainingState = new ClassifierTrainingState();

    @RequestMapping(value = "/generateCategoryClassifier", method = RequestMethod.POST)
    public void generateCategoryClassifier() throws IOException {
        if (!getCategoryClassifierTrainingState().isCurrentlyLearning()) {
            getCategoryClassifierTrainingState().setCurrentlyLearning(true);
            getService().generateCategoryClassifier(getCategoryClassifierTrainingState());
        }
    }

    @RequestMapping(value = "/generateBrandClassifier", method = RequestMethod.POST)
    public void generateBrandClassifier() throws IOException {
        if (!getBrandClassifierTrainingState().isCurrentlyLearning()) {
            getCategoryClassifierTrainingState().setCurrentlyLearning(true);
            getService().generateBrandClassifier(getBrandClassifierTrainingState());
        }
    }

    @RequestMapping(value = "/generateModel", method = RequestMethod.POST)
    public void generateModel() {
        if (!getModelTrainingState().isCurrentlyLearning()) {
            getCategoryClassifierTrainingState().setCurrentlyLearning(true);
            getService().generateModel(getModelTrainingState());
        }
    }

    @RequestMapping(value = "/generateAllClassifiers", method = RequestMethod.POST)
    public void generateAllClassifiers() throws IOException {
        getService().setTrainingAndTestingSet();
        generateCategoryClassifier();
        generateBrandClassifier();
        generateModel();
        getService().freeTestingSet();
    }
}
