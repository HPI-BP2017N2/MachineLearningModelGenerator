package de.hpi.modelgenerator.persistence.repo;

import de.hpi.modelgenerator.persistence.MatchingResult;
import lombok.Getter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Repository;

import javax.swing.table.TableRowSorter;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.springframework.data.mongodb.core.query.Criteria.where;
import static org.springframework.data.mongodb.core.query.Query.query;

@Repository
@Getter
public class MatchingResultRepository {

    @Autowired
    @Qualifier(value = "matchingResultTemplate")
    private MongoTemplate mongoTemplate;

    public Set<Long> getShopIds() {
        return getMongoTemplate().getCollectionNames().stream().map(Long::valueOf).collect(Collectors.toSet());
    }
    
    public List<MatchingResult> getMatches(long shopId, int count) {
        List<MatchingResult> matches = getMongoTemplate().find(query(where("offerKey").nin(null,"").and("matchingReason").is("ean").and("parsedData.title").nin("", null)).limit(count), MatchingResult.class, Long.toString(shopId));
        matches.addAll(getMongoTemplate().find(query(where("offerKey").nin(null,"").and("matchingReason").is("sku").and("parsedData.title").nin("", null)).limit(count), MatchingResult.class, Long.toString(shopId)));
        return matches;
        }
}
