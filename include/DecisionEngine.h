//the factory manager that makes the final decision 
// CNN says: "I'm 78% sure this is defective"
// DecisionEngine says: "Is 78% enough? Based on our rules... YES, REJECT!"

#ifndef DECISION_ENGINE_H
#define DECISION_ENGINE_H

#include <string> 
#include <vector> 
#include <map> 
#include <memory>

//enum decision 
enum class Decision {
    OK,
    DEFECT ,
    UNCERTAIN
};

//structure decision Result 
struct DecisionResult{
    Decision finalDecision;
    std::string label ;
    float defectProbabilities ;
    float confidence;
    bool overThreshold;
    std::string explanation ;

    DecisionResult();
    std::string toString() const;
    std::vector<int> getDisplayColor() const ;
};

struct DecisionConfig {
    float defectThreshold= 0.75f;

    float uncertaintyMargin = 0.10f;
    bool useUncertaintyThreshold = true ;

    //flag uncertain for human review 
    bool requireManualReview = false;
    //prob range for manual revieq 
    float manualReviewThreshold = 0.60f;

     // Cost-based parameters
    float costFalseNegative = 100.0f;     // Cost of missing a defect
    float costFalsePositive = 20.0f;      // Cost of false alarm
    bool useCostOptimization = false;     // Adjust threshold based on costs

    DecisionConfig();

    void validate() const; //validate config

    float calculateOptimalThreshold() const; //computes the optimal threshold based onthe cost

};

//decision engine class 
class DecisionEngine {
    public:
        DecisionEngine();
        explicit DecisionEngine(const DecisionConfig& config);

        virtual ~DecisionEngine();

        //cofig methods 
        void setConfig(const DecisionConfig& config);

        const DecisionConfig& getConfig() const;
        void setDefectThreshold(float newThreshold);

        void setUncertaintyZone(bool enabled);
        
        //main decision methods 

        DecisionResult makeDecision(float defectProbability);  //make a decision based on the defect proba

        DecisionResult makeDecisionWithcontext(
            float defectProbablity,
            const std::string& imageId= "",
            const std::map<std::string, std::string>& metadata={}
        );
        //process batch of proba 
        std::vector<DecisionResult> makeBatchDecision(
            const std::vector<float>& probabilities
        );

        //busniss rule methods
        bool isDefect(float probability) const ; //check if proba excces defect threshold

        //check if proba is in uncertainty zone
        bool requiresReview(float probability) const;

        //calculate decision confidence
        float calculateConfidence(float probability) const;

        std::string generateExplanation(float probability, Decision decision) const;

        //stats and monitpring
        std::map<std::string, int> getStatics() const;

        void resetStatics(); //reset statics counters
        void printConfig() const; //print current config

        static float calculateOptimalThreshold(
            float falseNegativeCost,
            float falsePositiveCost,
            float defectPrevalence = 0.05f
        );

    private:
        DecisionConfig config_;
        std::map<Decision , int> decisionCounts_;
        int totalDecisions_;

        //private helper emthods 
        

        //validate proba input
        void validateProbability(float probabilty) const ;

        //update Statistics counters
        void updateStatistics(Decision decision);

        Decision applyBusinessRules(
            float probabilty , Decision preliminaryDecision , const std::map<std::string, std::string>& metadata ={}
        );

        //check if manual review is required 
        bool requireManualReview(float probability) const;



};

#endif 