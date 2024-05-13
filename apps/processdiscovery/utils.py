from .models import ProcessDiscovery
from django.shortcuts import get_object_or_404
from json import JSONDecoder
from typing import Optional

###########################################################################################################################
# Trazablity between decision points and rules ############################################################################
###########################################################################################################################

## BRANCH

class Branch:
  """
  A branch is an activity or series of activities and decision points that are reached when the condition of a rule is satisfied
  """
  def __init__(self, label: str, decision_points: Optional[list['DecisionPoint']] = []):
    self.label = label
    if not decision_points is None:
      ids: list[str] = decision_points.map(lambda dp: dp.id)
      if len(ids) != len(set(ids)):
        raise ValueError("Decision points must have unique ids")
      self.decision_points = decision_points
  
  @staticmethod
  def from_json(json: dict) -> 'Branch':
    return BranchDecoder().decode(json)

class BranchDecoder(JSONDecoder):
  """
  A JSON decoder for Branch objects
  """
  def __init__(self):
    JSONDecoder.__init__(self, object_hook=self.dict_to_object)

  def dict_to_object(self, d: dict) -> Branch:
    # decode decision points into a list of DecisionPoint objects
    decision_points = d['decision_points'].map(lambda dp: DecisionPointDecoder().decode(dp))
    return Branch(d['label'], decision_points)

## RULE

class Rule:
  """
  A rule is a condition that is evaluated in a decision point. It has a condition in the form of logic operations and a target branch
  """
  def __init__(self, condition: str, target: str):
    self.condition = condition
    self.target = target
  
  @staticmethod
  def from_json(json: dict) -> 'Rule':
    return RuleDecoder().decode(json)

class RuleDecoder(JSONDecoder):
  """
  A JSON decoder for Rule objects
  """
  def __init__(self):
    JSONDecoder.__init__(self, object_hook=self.dict_to_object)

  def dict_to_object(self, d: dict) -> Rule:
    return Rule(d['condition'], d['target'])

## DECISION POINT

class DecisionPoint:
  """
  A decision point is a point in the process where a decision is made. It has a unique id, a previous activity, a list of branches and a list of rules
  """
  def __init__(self, id: str, prevAct: str, branches: list[Branch], rules: Optional[list[Rule]] = []):
    self.rules = rules
    labels = branches.map(lambda b: b.label)
    if len(labels) != len(set(labels)):
      raise ValueError("Branches must have unique labels")
    self.branches = branches
    self.prevAct = prevAct
    self.id = id

  @staticmethod
  def from_json(json: dict) -> 'DecisionPoint':
    return DecisionPointDecoder().decode(json)

class DecisionPointDecoder(JSONDecoder):
  """
  A JSON decoder for DecisionPoint objects
  """
  def __init__(self):
    JSONDecoder.__init__(self, object_hook=self.dict_to_object)

  def dict_to_object(self, d: dict) -> DecisionPoint:
    # decode branches into a list of Branch objects
    branches = d['branches'].map(lambda b: BranchDecoder().decode(b))
    # decode rules into a list of Rule objects
    rules = d['rules'].map(lambda r: RuleDecoder().decode(r))
    return DecisionPoint(d['id'], d['prevAct'], branches, rules)

## PROCESS

class Process:
  """
  A processes represented as a sequence of decision points. This is the root of the process
  """
  def __init__(self, decision_points: list[DecisionPoint]):
    ids = decision_points.map(lambda dp: dp.id)
    if len(ids) != len(set(ids)):
      raise ValueError("Decision points must have unique ids")
    self.decision_points = decision_points 

  @staticmethod
  def from_json(json: dict) -> 'Process':
    return ProcessDecoder().decode(json)
  
  # TODO: To Json

class ProcessDecoder(JSONDecoder):
  """
  A JSON decoder for Process objects
  """
  def __init__(self):
    JSONDecoder.__init__(self, object_hook=self.dict_to_object)

  def dict_to_object(self, d: dict) -> Process:
    # decode decision points into a list of DecisionPoint objects
    decision_points = d['decision_points'].map(lambda dp: DecisionPointDecoder().decode(dp))
    return Process(decision_points)

###########################################################################################################################
# case study get phases data  ###########################################################################################
###########################################################################################################################

def get_process_discovery(case_study):
  return get_object_or_404(ProcessDiscovery, case_study=case_study)

def case_study_has_process_discovery(case_study):
  return ProcessDiscovery.objects.filter(case_study=case_study, active=True).exists()
