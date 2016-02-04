#ifndef DIFACTO_LBFGS_LBFGS_LEARNER_H_
#define DIFACTO_LBFGS_LBFGS_LEARNER_H_
namespace difacto {

class LBFGSLearner : public Learner {
 public:
  virtual ~LBFGSLearner() { }

  KWArgs Init(const KWArgs& kwargs) override {}

 protected:
  void RunScheduler() override;

  void Process(const std::string& args, std::string* rets) override;

 private:

  void CalcGradient();

  void CalcDirection();

  void LinearSearch();
};
}  // namespace difacto
#endif  // DIFACTO_LBFGS_LBFGS_LEARNER_H_
